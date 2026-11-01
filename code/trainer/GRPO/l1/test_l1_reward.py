"""CPU logic tests with deterministic embeddings; no pretrained models or GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from l1 import prepare_data, verl_reward
import l1_reward as reward
import utils


class Matrix:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0]) if self.rows else 0

    @property
    def T(self):
        return Matrix(list(zip(*self.rows)))

    def __matmul__(self, other):
        return Matrix([[sum(a * b for a, b in zip(row, column)) for column in zip(*other.rows)]
                       for row in self.rows])

    def topk(self, k, dim):
        if dim != 1:
            raise ValueError("Test matrix supports topk only along columns")
        return SimpleNamespace(indices=Matrix([
            sorted(range(len(row)), key=lambda index: (-row[index], index))[:k] for row in self.rows
        ]))

    def tolist(self):
        return self.rows


class FakeEmbedder:
    def __init__(self):
        self.torch = SimpleNamespace(cat=lambda parts, dim: Matrix([
            sum((part.rows[index] for part in parts), []) for index in range(parts[0].shape[0])
        ]))
        self.clear_count = 0
        self.clear()

    def clear(self):
        self.clear_count += 1
        self.cache = {}
        self.calls = []
        self.computed = []

    def encode(self, texts, query):
        self.calls.append((query, list(texts)))
        for text in texts:
            key = query, text
            if key not in self.cache:
                vector = [0.0] * 66
                if query:
                    index = int(text.split(":")[0][1:])
                    vector[index if index < 25 else 64] = 1.0
                    vector[32 + index if index < 20 else 65] = 1.0
                elif text in ("PC", "FC"):
                    vector[0] = vector[32] = 4.0
                elif text in ("DP", "DF"):
                    vector[64 if text == "DP" else 65] = 1.0
                elif text.startswith(("P", "F")):
                    vector[int(text[1:]) + (32 if text.startswith("F") else 0)] = 1.0
                else:
                    raise ValueError(f"Unknown test embedding text: {text}")
                self.cache[key] = vector
                self.computed.append(key)
        return Matrix([self.cache[(query, text)] for text in texts]) if texts else None


def content(actions):
    return prepare_data.layer1.build_message([
        {"date": "2026-10-03", "source": "Bing", "action": action} for action in actions
    ])


def answer(count, evidence=None):
    return json.dumps({"interests": [
        {"interest_name": f"I{index}", "actual_activity": "Synthetic activity",
         "topics": [{"topic": "topic", "evidence": [0] if evidence is None else evidence}]}
        for index in range(count)
    ]})


def judge(prompt, text, finish_reason):
    output = reward.parse_json_object(text)
    valid = output is not None and utils.layer1_keys_valid(output)
    return ({"json_valid": valid, "rule_pass": valid and finish_reason != "length", "violations": []},
            output if valid else None)


class DualPoolRewardTest(unittest.TestCase):
    def setUp(self):
        self.language = patch.object(reward.user_profile_rules, "language_tools")
        self.judgement = patch.object(reward, "judge", side_effect=judge)
        self.language.start()
        self.judgement.start()
        self.addCleanup(self.language.stop)
        self.addCleanup(self.judgement.stop)
        self.embedder = FakeEmbedder()
        self.scorer = reward.L1Reward(embedder=self.embedder, top_k=1)
        self.past = [f"P{index}" for index in range(25)]
        self.future = [f"F{index}" for index in range(20)]
        self.scorer.set_pools(self.past + ["DP"], self.future + ["DF"])

    def score(self, count=1, evidence=None):
        return self.scorer(content(self.past), answer(count, evidence), None, self.future)

    def test_four_signal_mean_and_interest_cap(self):
        cap = reward.interest_cap(25)
        self.assertEqual(cap, 9)
        for count in (1, 5, 9, 12, 18, 27):
            with self.subTest(count=count):
                result = self.score(count)
                matched = min(count, cap)
                self.assertAlmostEqual(result["past_recall"], matched / 25)
                self.assertAlmostEqual(result["past_precision"], matched / count)
                self.assertAlmostEqual(result["future_recall"], matched / 20)
                self.assertAlmostEqual(result["future_precision"], matched / count)
                expected = (matched / 25 + matched / count + matched / 20 + matched / count) / 4
                self.assertAlmostEqual(result["reward"], expected)
                self.assertTrue(0 <= result["reward"] <= 1)

    def test_existing_cap_curve_is_unchanged(self):
        self.assertEqual([reward.interest_cap(n) for n in (1, 10, 25, 60, 200)], [1, 5, 9, 13, 17])

    def test_coverage_does_not_affect_reward(self):
        sparse = self.score(evidence=[0])
        dense = self.score(evidence=list(range(25)))
        self.assertEqual(sparse["coverage"], 1 / 25)
        self.assertEqual(dense["coverage"], 1)
        self.assertEqual(sparse["reward"], dense["reward"])
        for name in reward.REWARD_SIGNALS:
            self.assertEqual(sparse[name], dense[name])

    def test_pools_are_ranked_independently(self):
        original = self.score()
        self.scorer.set_pools(self.past + ["DP", "PC"], self.future + ["DF"])
        changed_past = self.score()
        self.assertEqual(changed_past["past_recall"], 0)
        self.assertEqual(changed_past["future_recall"], original["future_recall"])
        self.assertEqual(changed_past["future_precision"], original["future_precision"])
        self.scorer.set_pools(self.past + ["DP"], self.future + ["DF", "FC"])
        changed_future = self.score()
        self.assertEqual(changed_future["future_recall"], 0)
        self.assertEqual(changed_future["past_recall"], original["past_recall"])
        self.assertEqual(changed_future["past_precision"], original["past_precision"])

    def test_query_and_shared_action_embeddings_are_reused(self):
        before = self.embedder.clear_count
        self.scorer.set_pools(["P0", "P0"], ["P0", "F0", "F0"])
        self.assertEqual(self.embedder.clear_count, before + 1)
        self.assertEqual(self.scorer.past_pool, ["P0"])
        self.assertEqual(self.scorer.future_pool, ["P0", "F0"])
        self.assertEqual(self.embedder.computed.count((False, "P0")), 1)
        self.scorer(content(["P0"]), answer(1), None, ["F0"])
        self.assertEqual(sum(query for query, _ in self.embedder.calls), 1)

    def test_one_to_one_matching(self):
        self.assertEqual(reward.matched_interests([[[0, 1]], [[0, 1]]]), 1)
        self.assertEqual(reward.matched_interests([[[0, 1], [1, 2]], [[0, 1]]]), 2)

    def test_invalid_and_truncated_answers(self):
        invalid = self.scorer(content(self.past), "not JSON", None, self.future)
        self.assertEqual(invalid["reward"], 0)
        self.assertTrue(all(invalid[name] == 0 for name in reward.SIGNALS))
        truncated = self.scorer(content(self.past), answer(1), "length", self.future)
        self.assertEqual(truncated["reward"], 0)
        self.assertGreater(truncated["past_recall"], 0)
        self.assertGreater(truncated["future_recall"], 0)

    def test_empty_answer_and_future_targets(self):
        empty = self.score(0)
        self.assertEqual(empty["reward"], 0)
        self.assertEqual(empty["past_precision"], 0)
        no_future = self.scorer(content(self.past), answer(1), None, [])
        self.assertEqual(no_future["future_recall"], 0)
        self.assertEqual(no_future["future_precision"], 0)
        self.assertAlmostEqual(no_future["reward"], (1 / 25 + 1) / 4)

    def test_missing_past_pool_and_invalid_prompt_fail(self):
        self.scorer.set_pools([], self.future)
        with self.assertRaisesRegex(ValueError, "Past candidate pool is empty"):
            self.score()
        self.scorer.set_pools(["P0"], self.future)
        with self.assertRaisesRegex(ValueError, "batch/data alignment"):
            self.score()
        with self.assertRaisesRegex(ValueError, "Layer-1 prompt"):
            reward.input_signals("not a Layer-1 prompt")

    def test_adapter_returns_all_metrics_and_caches_both_pools(self):
        pools = {"past_actions": self.past + ["DP"], "future_actions": self.future + ["DF"]}
        with patch.object(verl_reward, "_state", {"reward_fn": self.scorer, "batch_id": None}), \
                patch.object(verl_reward, "load_pools", return_value=pools) as loader:
            for _ in range(2):
                result = verl_reward.compute_score("user_profile_l1", answer(1), self.future,
                                                  {"batch_id": "train-000000", "content": content(self.past)},
                                                  "pools.parquet")
            loader.assert_called_once_with("pools.parquet", "train-000000")
        self.assertEqual(set(result), {"score", "json_valid", "rule_pass", "n_interests", *reward.SIGNALS})
        self.assertTrue(all(isinstance(value, float) for value in result.values()))


class PoolPreparationTest(unittest.TestCase):
    def test_launcher_rejects_old_cache_before_other_preparation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "train.parquet").touch()
            (data / "meta.json").write_text(json.dumps({"batch_size": 64, "max_prompt_length": 17408}))
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "python3").symlink_to(sys.executable)
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "DATA_DIR": str(data), "OUTPUT_DIR": str(root / "output"), "LOGGER": "['console']",
                   "TRAIN_BATCH_SIZE": "64", "MAX_PROMPT_LENGTH": "17408", "MAX_RESPONSE_LENGTH": "8192",
                   "SFT_REPLAY": "1", "SFT_MAX_SEQ_LENGTH": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            launcher = Path(prepare_data.__file__).resolve().parents[1] / "run_l1_grpo.sh"
            result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected dual past/future pools", result.stderr)
        self.assertIn("regenerate", result.stderr)

    def test_past_pool_uses_only_visible_signals(self):
        past = [{"date": "2026-10-03", "source": "Bing", "action": action} for action in ("P0", "P1")]
        row = {"user_id": "synthetic-user", "past_behaviors": past,
               "future_behaviors": [{"date": "2026-10-10", "source": "Bing", "action": "F0"}]}
        with patch.object(prepare_data.layer1, "fit_prompt", return_value=(past[:1], [1, 2])):
            example = prepare_data.build_example(row, lambda text: [1], 10000)
        if example is None:
            raise AssertionError("The synthetic row should survive prompt preparation")
        pool = prepare_data.assign_batches([example], "train", 1)["train-000000"]
        self.assertEqual(pool, {"past_actions": ["P0"], "future_actions": ["F0"]})

    def test_batches_and_validation_pools(self):
        def row(past, future):
            return {"extra_info": {"content": content(past)}, "reward_model": {"ground_truth": future}}
        rows = [row(["P0"], ["F0"]), row(["P1"], ["F1"])]
        pools = prepare_data.assign_batches(rows, "train", 1)
        self.assertEqual(pools["train-000000"], {"past_actions": ["P0"], "future_actions": ["F0"]})
        self.assertEqual(pools["train-000001"], {"past_actions": ["P1"], "future_actions": ["F1"]})
        pools = prepare_data.assign_batches(rows, "test", None)
        self.assertEqual(pools["val"], {"past_actions": ["P0", "P1"], "future_actions": ["F0", "F1"]})

    def test_manifest_and_parquet_writer_columns(self):
        pa, pq = ModuleType("pyarrow"), ModuleType("pyarrow.parquet")
        pa.parquet = pq
        pa.Table = SimpleNamespace(from_pylist=lambda rows: rows, from_pydict=lambda values: values)
        pq.write_table = Mock()
        def row(past, future):
            return {"extra_info": {"content": content(past), "prompt_tokens": 2},
                    "reward_model": {"ground_truth": future}}
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.parquet": pq}):
            args = argparse.Namespace(output_dir=Path(directory), batch_size=1, seed=0,
                                      max_prompt_length=10000, tokenizer="synthetic")
            with patch("builtins.print"):
                prepare_data.write(args, [row(["P0"], ["F0"])], [row(["P1"], ["F1"])])
            meta = json.loads((Path(directory) / "meta.json").read_text())
        self.assertEqual(meta["pool_format_version"], prepare_data.POOL_FORMAT_VERSION)
        table = pq.write_table.call_args_list[2].args[0]
        self.assertEqual(table, {"batch_id": ["val", "train-000000"],
                                 "past_actions": [["P0"], ["P1"]], "future_actions": [["F0"], ["F1"]]})
        self.assertEqual(pq.write_table.call_args_list[2].kwargs, {"row_group_size": 1})


class CaseStudyTest(unittest.TestCase):
    def test_score_reports_both_sides(self):
        from case_study import case_study

        with TemporaryDirectory() as directory:
            root = Path(directory)
            signals = [{"date": "2026-10-03", "source": "Bing", "action": "P0"}]
            row = {"prompt_id": 0, "user_id": "synthetic", "content": content(["P0"]), "signals": signals,
                   "future": [{"date": "2026-10-10", "source": "Bing", "action": "F0"}],
                   "prompt_tokens": 2, "signals_dropped": 0,
                   "samples": [{"text": answer(1), "finish_reason": "stop"},
                               {"text": "not JSON", "finish_reason": "stop"}]}
            (root / "rollouts.jsonl").write_text(json.dumps(row) + "\n")
            args = argparse.Namespace(output_dir=root, embedding_model="synthetic", embedding_batch_size=256,
                                      top_k=1, embedding_url=None, report_prompts=1)
            with patch.object(reward.user_profile_rules, "language_tools"), \
                    patch.object(reward, "judge", side_effect=judge):
                scorer = reward.L1Reward(embedder=FakeEmbedder(), top_k=1)
                with patch.object(case_study, "L1Reward", return_value=scorer), patch("builtins.print"):
                    case_study.score(args)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["metrics"]["past_recall@1"]["mean"], 0.5)
            self.assertEqual(summary["metrics"]["past_precision@1"]["mean"], 0.5)
            self.assertEqual(summary["metrics"]["future_recall@1"]["mean"], 0.5)
            self.assertEqual(summary["metrics"]["future_precision@1"]["mean"], 0.5)
            self.assertIn("past_hits", (root / "report.html").read_text())

    def test_trend_reads_old_and_new_dumps(self):
        from case_study import case_study

        base = {"input": "synthetic prompt", "output": "synthetic answer", "score": 0.5,
                "json_valid": 1, "rule_pass": 1, "n_interests": 1,
                "coverage": 0.5, "future_recall": 0.5, "future_precision": 0.5}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, extra in (("old", {}), ("new", {"past_recall": 0.4, "past_precision": 0.6})):
                validation = root / name
                validation.mkdir()
                (validation / "0.jsonl").write_text(json.dumps({**base, **extra, "step": 0}) + "\n")
                output = root / f"{name}-output"
                with patch("builtins.print"):
                    case_study.trend(argparse.Namespace(validation_dir=validation, output_dir=output))
                step = json.loads((output / "trend.json").read_text())[0]
                self.assertEqual("past_recall" in step, name == "new")
                self.assertEqual(step["future_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
