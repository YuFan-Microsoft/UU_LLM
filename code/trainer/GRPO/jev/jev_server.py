"""JEV reward server: the Qwen3.5 profile-behavior classifier of ../../JEV/Qwen35_profile_classifier.py over HTTP.

Scores (profile, behavior) pairs exactly as JEV/Qwen35_profile_classifier_inference.py does: the trainer's prompt and
chat template, the profile-end truncation at --max_length, the checkpoint's frozen class head, right padding per batch.
One GPU: concurrent requests are pooled, sorted by length and packed into batches of at most --max_batch_tokens
padded tokens.

  GET  /health -> {"status": "ok", "service": "jev", "checkpoint": ...}   (the port opens once the model is loaded)
  POST /score  {"pairs": [{"profile": <str | list>, "behavior": <str | dict>}, ...]}
            -> {"results": [{"pred_label": "partial match",
                             "probability": {"no match": p, "partial match": p, "perfect match": p}}, ...]}
A string is used as is (the mai_profile / future_behavior JSON strings of EVAL_Dataset); a list or dict is serialized
as those were: json.dumps(ensure_ascii=False) without null keys.

  CUDA_VISIBLE_DEVICES=1 python3 jev_server.py --port 8200   # --checkpoint defaults to the reward model step9K
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Qwen35_profile_classifier.py: ../../JEV in the repository, or beside this file when serve_jev.sh copied both.
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "JEV"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from Qwen35_profile_classifier import (CLASS_HEAD_FILE, LABELS, FixedLengthCollator,  # noqa: E402
                                       ProfileBehaviorDataset, answer_token_ids, class_logits, load_classifier,
                                       require_linear_attention_kernels)


def drop_nulls(value):
    if isinstance(value, dict):
        return {key: drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value]
    return value


def to_text(value):
    return value if isinstance(value, str) else json.dumps(drop_nulls(value), ensure_ascii=False)


def pack(lengths, max_batch_tokens):
    """[start, end) ranges over ascending `lengths`: each batch is right-padded to its last (longest) row."""
    batches, start = [], 0
    for end in range(1, len(lengths) + 1):
        if end == len(lengths) or (end + 1 - start) * lengths[end] > max_batch_tokens:
            batches.append((start, end))
            start = end
    return batches


class Job:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.probs = [None] * len(input_ids)
        self.error = None
        self.done = threading.Event()


class Scorer:
    """Prompt encoding (in the request threads) and one GPU thread scoring every pending request together."""

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        self.tokenizer.padding_side = "right"
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.tokenizer.eos_token_id
        self.tokenizer_lock = threading.Lock()  # a fast tokenizer is not safe across threads
        self.max_length = args.max_length
        self.max_batch_tokens = args.max_batch_tokens
        self.device = torch.device(args.device)

        require_linear_attention_kernels()
        load_args = argparse.Namespace(attn_implementation=args.attn_implementation, gradient_checkpointing=0)
        self.model = load_classifier(args.checkpoint, answer_token_ids(self.tokenizer), load_args)
        self.model.to(self.device).eval()
        self.queue = queue.Queue()
        threading.Thread(target=self.run, daemon=True).start()

    def encode(self, profiles, behaviors):
        """Token ids of each pair, with the trainer's prompt and truncation."""
        data = (profiles, np.arange(len(profiles)), behaviors, np.zeros(len(profiles), dtype=np.int64))
        dataset = ProfileBehaviorDataset(data, self.tokenizer, self.max_length)
        with self.tokenizer_lock:
            return [dataset[i]["input_ids"] for i in range(len(dataset))]

    def score(self, input_ids):
        """[n, 3] softmax probabilities of LABELS; raises RuntimeError when the GPU round failed."""
        job = Job(input_ids)
        self.queue.put(job)
        job.done.wait()
        if job.error:
            raise RuntimeError(job.error)
        return job.probs

    def run(self):
        while True:
            jobs = [self.queue.get()]
            while True:
                try:
                    jobs.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            try:
                self.score_jobs(jobs)
            except Exception as error:  # e.g. CUDA OOM: fail this round's requests, keep serving
                for job in jobs:
                    job.error = f"{type(error).__name__}: {error}"
                torch.cuda.empty_cache()
            for job in jobs:
                job.done.set()

    @torch.inference_mode()
    def score_jobs(self, jobs):
        rows = sorted(((job, i) for job in jobs for i in range(len(job.input_ids))),
                      key=lambda row: len(row[0].input_ids[row[1]]))
        lengths = [len(job.input_ids[i]) for job, i in rows]
        for start, end in pack(lengths, self.max_batch_tokens):
            batch = FixedLengthCollator(self.pad_token_id, lengths[end - 1])(
                [{"input_ids": job.input_ids[i], "class_label": 0} for job, i in rows[start:end]])
            batch = {key: batch[key].to(self.device, non_blocking=True) for key in ("input_ids", "attention_mask")}
            probs = F.softmax(class_logits(self.model, batch), dim=-1).cpu().tolist()
            for (job, i), p in zip(rows[start:end], probs):
                job.probs[i] = p


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    scorer = None
    checkpoint = ""

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": f"unknown path {self.path}"})
            return
        self.reply(200, {"status": "ok", "service": "jev", "checkpoint": self.checkpoint})

    def do_POST(self):
        if self.path != "/score":
            self.reply(404, {"error": f"unknown path {self.path}"})
            return
        started = time.time()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pairs = body["pairs"]
            profiles = [to_text(pair["profile"]) for pair in pairs]
            behaviors = [to_text(pair["behavior"]) for pair in pairs]
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {"error": f'expected {{"pairs": [{{"profile": ..., "behavior": ...}}, ...]}}: {error!r}'})
            return
        try:
            probs = self.scorer.score(self.scorer.encode(profiles, behaviors)) if pairs else []
        except Exception as error:  # answer instead of dropping the connection, which the client would retry
            print(f"[jev] {len(pairs)} pairs failed: {error}", flush=True)
            self.reply(500, {"error": f"{type(error).__name__}: {error}"})
            return
        results = [{"pred_label": LABELS[int(np.argmax(p))],
                    "probability": {label: round(float(x), 6) for label, x in zip(LABELS, p)}} for p in probs]
        self.reply(200, {"results": results})
        print(f"[jev] {self.client_address[0]}: {len(pairs)} pairs in {time.time() - started:.1f}s", flush=True)

    def log_message(self, format, *args):  # per-request lines come from do_POST; /health polls stay quiet
        pass


def parse_args():
    parser = argparse.ArgumentParser("JEV profile-behavior classifier server")
    parser.add_argument("--checkpoint", default="/yufan/MAI_Profile/checkpoints/reward_model/step9K",
                        help=f"A step_<N> directory saved by Qwen35_profile_classifier.py (with {CLASS_HEAD_FILE})")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--device", default="cuda", help="Pick the GPU with CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Same as training: longer prompts have the end of the profile cut")
    parser.add_argument("--max_batch_tokens", type=int, default=65536,
                        help="Padded tokens per forward (rows x longest row); at least --max_length")
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Attention of the full-attention layers (flash_attention_2 or sdpa)")
    args = parser.parse_args()
    if args.max_batch_tokens < args.max_length:
        parser.error("--max_batch_tokens must be at least --max_length")
    if not (Path(args.checkpoint) / CLASS_HEAD_FILE).exists():
        parser.error(f"{args.checkpoint} has no {CLASS_HEAD_FILE}: not a Qwen35_profile_classifier.py checkpoint")
    return args


def main():
    args = parse_args()
    print(f"[jev] loading {args.checkpoint}", flush=True)
    scorer = Scorer(args)
    # Before the port opens, so /health answers only once scoring works: the largest batch a request can make
    # (rows of --max_length tokens filling --max_batch_tokens; an OOM shows up here, not mid-training), then one pair.
    scorer.score([[scorer.pad_token_id] * args.max_length] * (args.max_batch_tokens // args.max_length))
    warmup = scorer.score(scorer.encode(['[{"Interest": "Smartphones"}]'], ['{"Query": "galaxy s25 price"}']))
    print(f"[jev] warmup probability {[round(p, 4) for p in warmup[0]]}", flush=True)
    Handler.scorer, Handler.checkpoint = scorer, args.checkpoint
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[jev] serving on {args.host}:{args.port} (max_batch_tokens {args.max_batch_tokens})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
