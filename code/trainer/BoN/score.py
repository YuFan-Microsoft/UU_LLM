#!/usr/bin/env python3
"""Step 3 of best-of-N: every candidate of candidates.jsonl scored by JEV -> scores.jsonl.

The score of one candidate is the GRPO L3 reward (l3_reward.py) with one change: the profiled interests are the
candidate's OWN commercial labels, not the data's reference ones:
  1. 0 when the answer breaks the Layer-3 Commercial rules (user_profile_rules.score_example: not V3 JSON, interest
     names not exactly the input's, bad score / stage / query values, duplicates, query language; a cut answer is not
     valid JSON)
  2. else its profile: the input interests the answer marks "commercial": true with a commercial_score of at least
     --min_commercial_score (default medium, as the e2e ads eval), the first 10 in input order (the Layer-2
     confidence order), each with its Category (the row's Layer-3 Persona category), the answer's brands / products /
     retailers / predicted queries and the input topics, as the EVAL_Dataset mai_profile string JEV was trained on
     (jev_profile.build_mai_profile -> e2e_ads_eval/classify.mai_profile: same items, same order). No such interest:
     the empty profile "[]", scored like any other.
  3. the action_freq-weighted mean over the row's targets of the JEV expected match
     (P(partial) + 2 * P(perfect)) / 2, each target weighing 1 / (1 + action_freq / 100) (l1_reward.action_weight).
score_unweighted (every weight 1), the profile, n_profile_interests and the hack watches commercial_share /
entities_per_interest are kept for analysis. Each distinct (profile, target) pair of a chunk of rows is scored once,
the chunk in one JEVClient call spread over every server. Resumable: rows already in scores.jsonl are skipped.

  {"uid", "scores": [...], "candidates": [{"score", "score_unweighted", "rule_pass", "violations", "profile",
                                            "n_profile_interests", "commercial_share", "entities_per_interest"}]}

    python3 score.py --work_dir <dir> --jev_urls <serve_jev.sh's last line>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import statistics
import time

from bon_common import (COMMERCIAL_SCORE_RANK, DEFAULT_MIN_COMMERCIAL_SCORE, done_uids, dumps, read_jsonl)
from jev_profile import TOP_K, JEVClient, build_mai_profile, prompt_interests
from l1_reward import action_weight
from l3_reward import LABEL_WEIGHTS, hack_signals
import user_profile_rules
from utils import parse_json_object


def own_commercial(prompt: str, output: dict, min_commercial_score: str) -> list[str]:
    """The input interest names the answer marks commercial with a commercial_score >= min_commercial_score ("any": any
    score), in input order; the answer's first entry of a name counts, as in build_mai_profile."""
    entries = {}
    for entry in output["interest_commercial"]:
        entries.setdefault(entry["interest_name"], entry)
    names = []
    for interest in prompt_interests(prompt):
        entry = entries.get(interest["interest_name"])
        if entry is None or entry.get("commercial") is not True:
            continue
        score = entry.get("commercial_score")
        rank = COMMERCIAL_SCORE_RANK.get(score.strip().casefold(), 0) if isinstance(score, str) else 0
        if min_commercial_score == "any" or rank >= COMMERCIAL_SCORE_RANK[min_commercial_score]:
            names.append(interest["interest_name"])
    return names


def check_candidate(task: tuple[str, str, str, str]) -> dict:
    """Rule check + profile of one candidate (a process-pool task: fastText is not thread-safe)."""
    prompt, category, text, min_commercial_score = task
    rules = user_profile_rules.score_example([{"role": "user", "content": prompt}], text)
    result = {"rule_pass": rules["rule_pass"], "violations": rules["violations"], "profile": None,
              "n_profile_interests": 0, "commercial_share": None, "entities_per_interest": None}
    output = parse_json_object(text) if rules["json_valid"] else None
    if output is None:
        return result
    reference = own_commercial(prompt, output, min_commercial_score)
    result.update(hack_signals(prompt, output, reference))
    if rules["rule_pass"]:
        result["profile"] = build_mai_profile(prompt, category, text, reference=reference)
        result["n_profile_interests"] = min(len(reference), TOP_K)
    return result


def init_worker() -> None:
    user_profile_rules.language_tools()  # load fastText once per process


def score_chunk(client: JEVClient, rows: list[dict], candidates: list[dict], pool, min_commercial_score: str) -> list:
    tasks = [(row["prompt"], row["category"], c["text"], min_commercial_score)
             for row, line in zip(rows, candidates) for c in line["candidates"]]
    checked = list(pool.map(check_candidate, tasks, chunksize=16))
    pairs = list(dict.fromkeys((result["profile"], target) for result, row in
                               zip(checked, (row for row, line in zip(rows, candidates) for _ in line["candidates"]))
                               if result["profile"] is not None for target in row["targets"]))
    probs = dict(zip(pairs, client.score(pairs))) if pairs else {}
    out, position = [], 0
    for row, line in zip(rows, candidates):
        weights = [action_weight(freq) for freq in (row.get("target_freqs") or [None] * len(row["targets"]))]
        results = []
        for _ in line["candidates"]:
            result = checked[position]
            position += 1
            score = unweighted = 0.0
            if result["profile"] is not None and row["targets"]:
                matches = [sum(w * probs[(result["profile"], target)][label] for label, w in LABEL_WEIGHTS.items())
                           for target in row["targets"]]
                score = sum(w * m for w, m in zip(weights, matches)) / sum(weights)
                unweighted = statistics.mean(matches)
            results.append({"score": score, "score_unweighted": unweighted, **result})
        out.append({"uid": row["uid"], "scores": [r["score"] for r in results], "candidates": results})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--jev_urls", default=os.environ.get("JEV_URLS"), help="serve_jev.sh's last line")
    parser.add_argument("--min_commercial_score", choices=["any", *COMMERCIAL_SCORE_RANK],
                        default=DEFAULT_MIN_COMMERCIAL_SCORE)
    parser.add_argument("--chunk_rows", type=int, default=64, help="Rows per JEV call")
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="Processes for the rule checks")
    args = parser.parse_args()
    if not args.jev_urls:
        parser.error("--jev_urls (or JEV_URLS) is required")

    from pathlib import Path
    work = Path(args.work_dir)
    out = work / "scores.jsonl"
    rows = {row["uid"]: row for row in read_jsonl(work / "rows.jsonl")}
    candidates = read_jsonl(work / "candidates.jsonl")
    done = done_uids(out)
    todo = [line for line in candidates if line["uid"] not in done]
    print(f"{len(todo)} rows to score ({len(done)} done)", flush=True)
    client = JEVClient(args.jev_urls)
    started = time.time()
    with ProcessPoolExecutor(args.num_workers, initializer=init_worker) as pool, out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.chunk_rows):
            chunk = todo[start:start + args.chunk_rows]
            for line in score_chunk(client, [rows[c["uid"]] for c in chunk], chunk, pool, args.min_commercial_score):
                f.write(dumps(line) + "\n")
            f.flush()
            finished = start + len(chunk)
            print(f"  {finished}/{len(todo)} rows ({(time.time() - started) / finished:.2f}s/row)", flush=True)
    (work / "score_meta.json").write_text(json.dumps({"min_commercial_score": args.min_commercial_score,
                                                      "jev_urls": args.jev_urls}, indent=2) + "\n")


if __name__ == "__main__":
    main()
