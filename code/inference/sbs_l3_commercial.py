#!/usr/bin/env python3
"""Side by side Layer-3 Commercial answers: the reference answer of SFT_User_Profile_L3_Commercial_gpt56_V3 (response_1,
GPT-5.6) and a checkpoint's answer to the same prompt (response_2), for analysis.

Each row's user message (the Layer-3 Commercial prompt) is answered by --checkpoint exactly as this folder's inference
answers it (layer3.COMMERCIAL's stage and key check, utils.generate with temperature 0.6 / top-p 0.8, up to 2
regenerations of an answer that is not valid, the run_inference.py vLLM engine setting); one vLLM engine per visible
GPU, the rows dealt round-robin. Both answers are scored by the SFT rollout checks (user_profile_rules.score_example).

    <output_dir>/sbs.jsonl      one input per line, in dataset order:
        {"uid", "prompt", "response_1", "response_2", "rules_1", "rules_2" ({json_valid, rule_pass, violations}),
         "response_2_attempts",
         "interests": [{"interest_name", "response_1": entry | null, "response_2": entry | null}]}
        "interests" lists the input interests in input order with each answer's entry of that name (commercial,
        commercial_score, intent_funnel_stage, brands, retailers, products, predicted_queries); null when the answer
        is not valid JSON or has no entry of that name
    <output_dir>/summary.json   per answer: json_valid / rule_pass ratios, commercial share of the input interests,
        commercial_score counts, mean brands + retailers + products and predicted queries per interest (commercial
        and not), the share of commercial interests without an entity / without a query; and the agreement of the
        two answers' commercial flags over the interests both answer

    python sbs_l3_commercial.py --checkpoint <ckpt_dir> --output_dir <dir> [--split test] [--max_rows N]
"""

import argparse
from collections import Counter
import json
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import statistics
import traceback

from utils import INPUT_MARKER, visible_gpus

HF_REPO = "yufan/user_profile_dataset"
HF_DIR = "User_Profile_L3_Commercial_gpt56_V3"  # the folder of the config SFT_User_Profile_L3_Commercial_gpt56_V3
ENTITY_KEYS = ("brands", "retailers", "products")


def load_rows(args: argparse.Namespace) -> list[dict]:
    """[{"uid", "prompt", "response_1"}] of the split, in dataset order (only its parquet shards are downloaded)."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(HF_REPO, repo_type="dataset", token=os.getenv("HF_TOKEN"),
                                  allow_patterns=[f"{HF_DIR}/{args.split}-*.parquet"]))
    rows = []
    for path in sorted((root / HF_DIR).glob(f"{args.split}-*.parquet")):
        for row in pq.read_table(path, columns=["uid", "messages"]).to_pylist():
            messages = {message["role"]: message["content"] for message in row["messages"]}
            rows.append({"uid": row["uid"], "prompt": messages["user"], "response_1": messages["assistant"]})
            if args.max_rows and len(rows) == args.max_rows:
                return rows
    if not rows:
        raise SystemExit(f"No {HF_DIR}/{args.split}-*.parquet rows")
    return rows


def engine(rank: int, gpu: str, args: argparse.Namespace, mine: list[tuple[int, dict]], out_path: Path) -> None:
    """One vLLM engine: answers its (row index, row) pairs into out_path (JSON lines)."""
    from transformers import AutoProcessor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm import LLM

    from layer3 import COMMERCIAL
    from run_inference import engine_kwargs
    from utils import MAX_MODEL_LEN, build_prompt_ids, generate, new_item

    try:
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):  # transformers 5 + vLLM 0.24
            PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
        tokenizer = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True).tokenizer
        llm = LLM(model=args.checkpoint, hf_overrides={"architectures": ["Qwen3_5ForConditionalGeneration"]},
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=args.seed + rank, trust_remote_code=True,
                  disable_log_stats=True, **engine_kwargs(args))
        items = [{**new_item(row["uid"], build_prompt_ids(tokenizer, [{"role": "user", "content": row["prompt"]}])),
                  "stage": COMMERCIAL.stage, "check": COMMERCIAL.keys_valid} for _, row in mine]
        generate(llm, items, None, COMMERCIAL.stage, temperature=args.temperature, top_p=args.top_p,
                 max_retries=args.max_retries, seed=args.seed)
        with out_path.open("w", encoding="utf-8") as out:
            for (index, _), item in zip(mine, items):
                out.write(json.dumps({"index": index, "text": item["text"], "attempts": item["attempts"]},
                                     ensure_ascii=False) + "\n")
        print(f"[engine {rank}] {len(items)} answers", flush=True)
    except BaseException:
        print(f"\n===== engine {rank} (GPU {gpu}) failed =====\n{traceback.format_exc()}"
              f"===== end of engine {rank} error =====\n", flush=True)
        raise


def run_engines(args: argparse.Namespace, rows: list[dict], tmp_dir: Path) -> dict[int, dict]:
    """{row index: {"text", "attempts"}} of the checkpoint's answers, one spawned engine per GPU (fail fast)."""
    from run_inference import DEFAULT_VLLM_PORT_BASE, PORTS_PER_ENGINE

    all_gpus = visible_gpus()
    gpus = all_gpus[:len(rows)]
    context = mp.get_context("spawn")
    original_port = os.environ.get("VLLM_PORT")
    port_base = int(original_port) if original_port and original_port.isdigit() else DEFAULT_VLLM_PORT_BASE
    processes = []
    for rank, gpu in enumerate(gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # a spawned process inherits the environment at start()
        os.environ["VLLM_PORT"] = str(port_base + rank * PORTS_PER_ENGINE)  # engines must not pick the same port
        mine = [(index, row) for index, row in enumerate(rows) if index % len(gpus) == rank]  # only its own rows
        process = context.Process(target=engine, args=(rank, gpu, args, mine, tmp_dir / f"{rank}.jsonl"))
        process.start()
        processes.append(process)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus)
    if original_port is None:
        os.environ.pop("VLLM_PORT")
    else:
        os.environ["VLLM_PORT"] = original_port
    while any(process.is_alive() for process in processes):
        mp.connection.wait([process.sentinel for process in processes if process.is_alive()])
        if any(process.exitcode not in (None, 0) for process in processes):  # fail fast: stop the other engines
            for process in processes:
                if process.is_alive():
                    process.terminate()
    for process in processes:
        process.join()
    errors = [(rank, process.exitcode) for rank, process in enumerate(processes) if process.exitcode != 0]
    if errors:
        raise RuntimeError(f"Engine(s) failed: {errors}; tracebacks above ('===== engine N ... failed =====')")
    answers = {}
    for rank in range(len(gpus)):
        with (tmp_dir / f"{rank}.jsonl").open(encoding="utf-8") as shard:
            for line in shard:
                answer = json.loads(line)
                answers[answer.pop("index")] = answer
    return answers


def entries(output: dict | None) -> dict[str, dict]:
    """{interest_name: the answer's first entry of that name} of a key-valid answer, else {}."""
    found = {}
    for entry in (output or {}).get("interest_commercial", []):
        found.setdefault(entry["interest_name"], entry)
    return found


def compare(row: dict, text_2: str, attempts: int) -> dict:
    """One sbs.jsonl record."""
    import user_profile_rules
    from layer3 import COMMERCIAL
    from utils import parse_json_object

    names = [interest["interest_name"] for interest in json.loads(row["prompt"].split(INPUT_MARKER, 1)[1])["interests"]]
    record = {"uid": row["uid"], "prompt": row["prompt"], "response_1": row["response_1"], "response_2": text_2,
              "response_2_attempts": attempts}
    found = []
    for side, text in (("1", row["response_1"]), ("2", text_2)):
        rules = user_profile_rules.score_example([{"role": "user", "content": row["prompt"]}], text)
        record[f"rules_{side}"] = {key: rules[key] for key in ("json_valid", "rule_pass", "violations")}
        output = parse_json_object(text)
        found.append(entries(output if output is not None and COMMERCIAL.keys_valid(output) else None))
    record["interests"] = [{"interest_name": name, "response_1": found[0].get(name), "response_2": found[1].get(name)}
                           for name in names]
    return record


def summarize(records: list[dict]) -> dict:
    """summary.json (see the module docstring)."""
    def mean(values):
        values = list(values)
        return round(statistics.mean(values), 4) if values else None

    summary = {"rows": len(records)}
    for side in ("1", "2"):
        key = f"response_{side}"
        answered = [i[key] for r in records for i in r["interests"] if i[key] is not None]
        commercial = [entry for entry in answered if entry["commercial"] is True]
        other = [entry for entry in answered if entry["commercial"] is not True]

        def entity_count(entry):
            return sum(len(entry[k]) for k in ENTITY_KEYS)

        summary[key] = {
            "json_valid": mean(r[f"rules_{side}"]["json_valid"] for r in records),
            "rule_pass": mean(r[f"rules_{side}"]["rule_pass"] for r in records),
            "violations": dict(Counter(v for r in records for v in r[f"rules_{side}"]["violations"]).most_common()),
            "interests_answered": len(answered),
            "commercial_share": mean(entry["commercial"] is True for entry in answered),
            "commercial_score": dict(Counter(str(entry["commercial_score"]) for entry in commercial)),
            "entities_per_commercial_interest": mean(map(entity_count, commercial)),
            "entities_per_other_interest": mean(map(entity_count, other)),
            "queries_per_commercial_interest": mean(len(entry["predicted_queries"]) for entry in commercial),
            "queries_per_other_interest": mean(len(entry["predicted_queries"]) for entry in other),
            "commercial_without_entity": mean(entity_count(entry) == 0 for entry in commercial),
            "commercial_without_query": mean(not entry["predicted_queries"] for entry in commercial),
        }
    both = [i for r in records for i in r["interests"] if i["response_1"] is not None and i["response_2"] is not None]
    pairs = Counter((i["response_1"]["commercial"] is True, i["response_2"]["commercial"] is True) for i in both)
    summary["commercial_flags"] = {
        "interests_both_answered": len(both),
        "agreement": mean(a == b for a, b in ((i["response_1"]["commercial"] is True,
                                               i["response_2"]["commercial"] is True) for i in both)),
        "both_commercial": pairs[(True, True)], "only_response_1": pairs[(True, False)],
        "only_response_2": pairs[(False, True)], "neither": pairs[(False, False)],
    }
    return summary


def parse_args() -> argparse.Namespace:
    from run_inference import ENGINE_PRESETS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="The model of response_2")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="test", help="test (4998 rows) or train")
    parser.add_argument("--max_rows", type=int, default=0, help="Only the first N rows (0 = all)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_retries", type=int, default=2, help="Regenerations of an invalid answer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine_preset", choices=list(ENGINE_PRESETS), default="original")
    args = parser.parse_args()
    for key, value in ENGINE_PRESETS[args.engine_preset].items():
        setattr(args, key, value)
    return args


def main() -> None:
    import shutil
    import tempfile

    from run_inference import prepare_language_tools

    args = parse_args()
    prepare_language_tools()  # the rule checks' fastText model, before hours of generation
    rows = load_rows(args)
    print(f"{len(rows)} {HF_DIR}/{args.split} rows; {args.checkpoint} on GPUs {visible_gpus()}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sbs_l3_", dir=args.output_dir))
    try:
        answers = run_engines(args, rows, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    records = [compare(row, answers[index]["text"], answers[index]["attempts"]) for index, row in enumerate(rows)]
    with (args.output_dir / "sbs.jsonl").open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"checkpoint": args.checkpoint, "split": args.split,
               "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_retries": args.max_retries,
                            "seed": args.seed, "engine_preset": args.engine_preset},
               **summarize(records)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {args.output_dir}/sbs.jsonl and summary.json", flush=True)


if __name__ == "__main__":
    main()
