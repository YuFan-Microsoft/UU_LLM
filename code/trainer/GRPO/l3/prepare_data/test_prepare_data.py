"""CPU tests for monthly preparation; no checkpoint downloads or GPU inference."""

from datetime import date, timedelta
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from l3.prepare_data import data_pipeline as pipeline, prepare_data


def monthly_row(user="user", cutoff="2026-11-01"):
    end = date.fromisoformat(cutoff)
    return {
        "user_id": user, "cutoff": cutoff,
        "past_behaviors": [
            {"date": (end - timedelta(days=28 - week * 7)).isoformat(),
             "source": "Bing", "action": f"past-{week}"}
            for week in range(4)
        ],
        "future_behaviors": [{"date": cutoff, "source": "Bing", "action": "FUTURE_ONLY"}],
    }


class FakeInference:
    def __init__(self, fail_stage=None, truncate_stage=None):
        self.prompts = {}
        self.calls = []
        self.fail_stage = fail_stage
        self.truncate_stage = truncate_stage

    def encode(self, content):
        key = len(self.prompts) + 1
        self.prompts[key] = content
        return [key]

    def answer(self, content):
        payload = content.split("\nInput:\n", 1)[1]
        if '"snapshot":' in payload:
            data = json.loads(payload)
            return {"decisions": [
                {"action": "add", "delta_interest_name": interest["interest_name"],
                 "actual_activity": interest["actual_activity"]}
                for interest in data["delta"]
            ]}
        signal = json.loads(payload.splitlines()[0])
        return {"interests": [{"interest_name": f"Interest {signal['date']}", "actual_activity": signal["action"],
                               "topics": [{"topic": "Synthetic topic", "evidence": [0]}]}]}

    def run(self, items, validator, stage):
        self.calls.extend((stage, item["key"], self.prompts[item["prompt_ids"][0]]) for item in items)
        for item in items:
            item["attempts"] = 1
            item["finish_reason"] = "length" if stage == self.truncate_stage else "stop"
            if stage != self.fail_stage:
                item["output"] = self.answer(self.prompts[item["prompt_ids"][0]])
                if not validator(item["output"]):
                    raise AssertionError("Fake answer must satisfy the real inference schema")


def prepared(user="user"):
    record = pipeline.parse_record(monthly_row(user), "synthetic.jsonl", 1)
    fake = FakeInference()
    snapshots, rejected, _ = pipeline.build_profiles([record], fake.encode, fake.run)
    if rejected:
        raise AssertionError(rejected)
    row, reason = pipeline.make_example(record, snapshots[user], fake.encode, 100)
    if reason is not None or row is None:
        raise AssertionError(reason)
    return row


class InputTest(unittest.TestCase):
    def test_explicit_cutoff_required(self):
        raw = monthly_row()
        del raw["cutoff"]
        with self.assertRaisesRegex(ValueError, "explicit"):
            pipeline.parse_record(raw, "input", 1)

    def test_half_open_boundaries(self):
        for field, day in (("past_behaviors", "2026-11-01"),
                           ("past_behaviors", "2026-10-03"),
                           ("future_behaviors", "2026-10-31"),
                           ("future_behaviors", "2026-11-08")):
            with self.subTest(field=field, day=day):
                raw = monthly_row()
                raw[field][0]["date"] = day
                with self.assertRaisesRegex(ValueError, "outside"):
                    pipeline.parse_record(raw, "input", 1)

    def test_timestamp_timezone_and_boundary_metadata(self):
        raw = monthly_row()
        raw["future_behaviors"][0]["date"] = "2026-10-31T23:30:00-02:00"
        record = pipeline.parse_record(raw, "input", 1)
        self.assertEqual(record.future[0]["date"], "2026-11-01")
        raw["history_start"] = "2026-10-01"
        with self.assertRaisesRegex(ValueError, "history_start"):
            pipeline.parse_record(raw, "input", 1)

    def test_cleaning_preserves_weekly_recurrence_and_future_overlap(self):
        raw = monthly_row()
        for item in raw["past_behaviors"] + raw["future_behaviors"]:
            item["action"] = "repeated action"
        raw["past_behaviors"].append({"date": "2026-10-05", "source": "Unknown", "action": "ignored"})
        record = pipeline.parse_record(raw, "input", 1)
        self.assertEqual(len(record.past), 4)
        self.assertEqual(record.future[0]["action"], "repeated action")
        fake = FakeInference()
        pipeline.build_profiles([record], fake.encode, fake.run)
        self.assertEqual(sum(stage == "l1" for stage, _, _ in fake.calls), 4)

    def test_duplicate_and_cross_split_users_fail_preflight(self):
        for duplicate_split in ("train", "test"):
            with self.subTest(split=duplicate_split), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "train.jsonl").write_text(json.dumps(monthly_row("same")) + "\n", encoding="utf-8")
                (root / "test.jsonl").write_text(json.dumps(monthly_row("other")) + "\n", encoding="utf-8")
                with (root / f"{duplicate_split}.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(monthly_row("same")) + "\n")
                with self.assertRaisesRegex(ValueError, "Duplicate user"):
                    prepare_data.index_inputs(root, 0)

    def test_index_order_is_seeded_and_readable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                (root / f"{split}.jsonl").write_text(
                    "".join(json.dumps(monthly_row(f"{split}-{i}")) + "\n" for i in range(5)), encoding="utf-8")
            locations, sources = prepare_data.index_inputs(root, 4)
            again, _ = prepare_data.index_inputs(root, 4)
            self.assertEqual(locations, again)
            self.assertEqual(len(sources), 2)
            self.assertEqual([item.index for item in locations], list(range(10)))
            for location in locations:
                self.assertEqual(prepare_data.load_location(location).user_id, location.user_id)

    def test_debug_local_limit_stops_before_unused_rows_and_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-a.jsonl").write_text(json.dumps(monthly_row("first")) + "\n", encoding="utf-8")
            prefix = "".join(json.dumps(monthly_row(user)) + "\n" for user in ("second", "third"))
            (root / "train-b.jsonl").write_bytes((prefix + "INVALID UNUSED ROW\n").encode("utf-8"))
            (root / "train-c.jsonl").write_text("INVALID UNUSED SHARD\n", encoding="utf-8")
            (root / "test.jsonl").write_text("INVALID UNUSED TEST DATA\n", encoding="utf-8")
            locations, sources = prepare_data.index_inputs(root, 0, max_users=3)
            self.assertEqual({item.user_id for item in locations}, {"first", "second", "third"})
            self.assertTrue(all(item.split == "train" for item in locations))
            self.assertEqual(len(sources), 2)
            self.assertEqual(sources[1]["hashed_bytes"], len(prefix.encode("utf-8")))
            self.assertTrue(all(source["hash_scope"] == "read_prefix" for source in sources))


class PreparationTest(unittest.TestCase):
    def test_own_calendar_no_future_and_no_l3_generation(self):
        records = [pipeline.parse_record(monthly_row("a", "2026-11-01"), "input", 1),
                   pipeline.parse_record(monthly_row("b", "2026-10-18"), "input", 2)]
        fake = FakeInference()
        snapshots, rejected, calls = pipeline.build_profiles(records, fake.encode, fake.run)
        self.assertEqual(rejected, {})
        self.assertEqual(snapshots["a"]["date"], "20261031")
        self.assertEqual(snapshots["b"]["date"], "20261017")
        self.assertEqual(sum(stage == "l1" for stage, _, _ in fake.calls), 8)
        self.assertEqual(sum(stage == "l2" for stage, _, _ in fake.calls), 6)
        self.assertEqual(set(stage for stage, _, _ in fake.calls), {"l1", "l2"})
        self.assertTrue(all("FUTURE_ONLY" not in content for _, _, content in fake.calls))
        for record in records:
            row, reason = pipeline.make_example(record, snapshots[record.user_id], fake.encode, 100)
            self.assertIsNone(reason)
            self.assertIsNotNone(row)
            self.assertEqual(row["reward_model"]["ground_truth"], ["FUTURE_ONLY"])
            self.assertNotIn("FUTURE_ONLY", row["prompt"][0]["content"])
            payload = pipeline.layer3.COMMERCIAL.payload(pipeline.layer3.active_interests(snapshots[record.user_id]))
            self.assertEqual(row["prompt"][0]["content"], pipeline.layer3.COMMERCIAL.message(payload))
            self.assertEqual(len(row["extra_info"]["past_actions"]), 4)
            self.assertEqual(len(calls[record.user_id]), 7)

    def test_failed_or_truncated_upstream_is_not_backfilled(self):
        record = pipeline.parse_record(monthly_row(), "input", 1)
        for stage in ("l1", "l2"):
            for mode in ("failure", "truncation"):
                with self.subTest(stage=stage, mode=mode):
                    fake = FakeInference(fail_stage=stage if mode == "failure" else None,
                                         truncate_stage=stage if mode == "truncation" else None)
                    _, rejected, _ = pipeline.build_profiles([record], fake.encode, fake.run)
                    self.assertEqual(rejected, {"user": f"{stage}_generation_failed"})

    def test_idle_last_week_carries_snapshot_without_extending_history(self):
        raw = monthly_row()
        raw["past_behaviors"] = raw["past_behaviors"][:3]
        record = pipeline.parse_record(raw, "input", 1)
        fake = FakeInference()
        snapshots, _, _ = pipeline.build_profiles([record], fake.encode, fake.run)
        self.assertEqual(snapshots["user"]["date"], "20261024")
        self.assertTrue(snapshots["user"]["_carried_forward"])

    def test_final_prompt_is_discarded_not_trimmed(self):
        record = pipeline.parse_record(monthly_row(), "input", 1)
        fake = FakeInference()
        snapshots, _, _ = pipeline.build_profiles([record], fake.encode, fake.run)
        snapshot = snapshots["user"]
        self.assertIsNone(pipeline.make_example(record, snapshot, lambda text: [0] * 5, 5)[1])
        self.assertEqual(pipeline.make_example(record, snapshot, lambda text: [0] * 6, 5),
                         (None, "prompt_too_long"))
        interest = snapshot["interests"][0]
        snapshot = {"interests": [{**interest, "interest_name": str(i)} for i in range(31)]}
        self.assertEqual(pipeline.make_example(record, snapshot, fake.encode, 100), (None, "too_many_interests"))
        self.assertEqual(pipeline.make_example(record, {}, fake.encode, 100), (None, "empty_profile"))

    def test_worker_with_fake_engine_runs_real_generation_helper(self):
        fake = FakeInference()
        tokenizer = SimpleNamespace(apply_chat_template=lambda *, conversation, **kwargs:
                                    fake.encode(conversation[0]["content"]))
        transformers, base, vllm = (ModuleType(name) for name in
                                    ("transformers", "transformers.tokenization_utils_base", "vllm"))
        transformers.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(
            tokenizer=tokenizer)))
        base.PreTrainedTokenizerBase = type("FakeTokenizerBase", (), {})

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, params, use_tqdm=False):
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text=json.dumps(fake.answer(fake.prompts[prompt["prompt_token_ids"][0]])),
                    finish_reason="stop")]) for prompt in prompts]

        vllm.LLM = FakeLLM
        vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "test"):
                (root / f"{split}.jsonl").write_text(json.dumps(monthly_row(split)) + "\n", encoding="utf-8")
            locations, _ = prepare_data.index_inputs(root, 0)
            args = prepare_data.parse_args(["--data_dir", str(root), "--output_dir", str(root / "output"),
                                           "--checkpoint", "synthetic"])
            modules = {"transformers": transformers, "transformers.tokenization_utils_base": base, "vllm": vllm}
            with patch.dict(sys.modules, modules), patch("builtins.print"):
                prepare_data.worker(0, args, locations, root / "worker.jsonl")
            events = [json.loads(line) for line in (root / "worker.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event["reason"] is None for event in events))
            self.assertTrue(all(event["row"]["data_source"] == pipeline.DATA_SOURCE for event in events))
            self.assertTrue(all("FUTURE_ONLY" not in text for text in fake.prompts.values()))


class ExportTest(unittest.TestCase):
    def test_direct_script_entry_and_default_output_directories(self):
        args = prepare_data.parse_args([])
        self.assertEqual(args.output_dir, prepare_data.GRPO_DIR / "outputs" / "l3_monthly")
        args = prepare_data.parse_args(["--max_users", "10"])
        self.assertEqual(args.output_dir, prepare_data.GRPO_DIR / "outputs" / "l3_debug")
        args = prepare_data.parse_args(["--max_users", "10", "--output_dir", "custom_output"])
        self.assertEqual(args.output_dir, Path("custom_output"))
        result = subprocess.run([sys.executable, "-B", "prepare_data.py", "--help"],
                                cwd=Path(prepare_data.__file__).parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max_users", result.stdout)

    def test_real_parquet_pools_split_isolation_and_partial_batches(self):
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            events = [{"index": i, "split": "train" if i < 4 else "test", "user_id": str(i),
                       "source_file": "synthetic", "source_line": i + 1,
                       "reason": "prompt_too_long" if i == 2 else None,
                       "row": None if i == 2 else prepared(str(i))} for i in range(5)]
            for event in events:
                if event["row"]:
                    event["row"]["extra_info"]["past_actions"] = [f"past-{event['user_id']}"]
                    event["row"]["reward_model"]["ground_truth"] = [f"future-{event['user_id']}"]
            paths = [root / "worker-0.jsonl", root / "worker-1.jsonl"]
            for rank, path in enumerate(paths):
                path.write_text("".join(json.dumps(event) + "\n" for event in events[rank::2]), encoding="utf-8")
            args = prepare_data.parse_args(["--data_dir", str(root), "--output_dir", str(root),
                                           "--batch_size", "2", "--checkpoint", "synthetic"])
            meta = prepare_data.write_outputs(args, paths, root, [])
            train = pq.read_table(root / "train.parquet").to_pylist()
            test = pq.read_table(root / "test.parquet").to_pylist()
            pools = pq.read_table(root / "pools.parquet").to_pylist()
            self.assertEqual([row["extra_info"]["user_id"] for row in train], ["0", "1", "3"])
            self.assertEqual([row["extra_info"]["user_id"] for row in test], ["4"])
            self.assertEqual([row["extra_info"]["batch_id"] for row in train],
                             ["train-000000", "train-000000", "train-000001"])
            by_id = {pool["batch_id"]: pool for pool in pools}
            self.assertEqual(by_id["test-000000"]["past_actions"], ["past-4"])
            self.assertEqual(by_id["train-000001"]["future_actions"], ["future-3"])
            self.assertEqual(meta["splits"]["train"]["last_batch_size"], 1)
            self.assertEqual(meta["splits"]["train"]["discarded"], {"prompt_too_long": 1})
            self.assertEqual(len((root / "discarded.jsonl").read_text().splitlines()), 1)
            self.assertEqual(pq.ParquetFile(root / "pools.parquet").metadata.num_row_groups, 3)

    def test_invalid_cli_limits(self):
        for flag, value in (("--batch_size", "0"), ("--max_prompt_length", "17409"),
                            ("--max_retries", "-1"), ("--top_p", "0"), ("--max_users", "-1")):
            with self.subTest(flag=flag), patch("sys.stderr"), self.assertRaises(SystemExit):
                prepare_data.parse_args(["--data_dir", "input", "--output_dir", "output", flag, value])

    def test_debug_main_publishes_only_train_even_when_sample_is_discarded(self):
        import pyarrow.parquet as pq

        for discard in (False, True):
            with self.subTest(discard=discard), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "train.jsonl").write_text(
                    "".join(json.dumps(monthly_row(f"user-{i}")) + "\n" for i in range(3)), encoding="utf-8")
                (root / "test.jsonl").write_text("DO NOT READ\n", encoding="utf-8")
                output = root / "output"
                args = prepare_data.parse_args(["--data_dir", str(root), "--output_dir", str(output),
                                               "--max_users", "1", "--checkpoint", "synthetic"])

                def fake_workers(settings, locations, temporary):
                    self.assertEqual(len(locations), 1)
                    self.assertEqual(locations[0].user_id, "user-0")
                    self.assertEqual(locations[0].split, "train")
                    path = temporary / "worker-0.jsonl"
                    event = {"index": 0, "split": "train", "user_id": "user-0", "source_file": "synthetic",
                             "source_line": 1, "reason": "prompt_too_long" if discard else None,
                             "row": None if discard else prepared("user-0")}
                    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
                    return [path]

                with patch.object(prepare_data, "parse_args", return_value=args), \
                        patch.object(prepare_data, "download_hf_inputs") as download, \
                        patch.object(prepare_data, "run_workers", side_effect=fake_workers), patch("builtins.print"):
                    prepare_data.main()
                download.assert_not_called()
                self.assertEqual({path.name for path in output.iterdir()},
                                 {"train.parquet", "pools.parquet", "discarded.jsonl", "meta.json"})
                meta = json.loads((output / "meta.json").read_text())
                self.assertTrue(meta["debug"])
                self.assertEqual(meta["max_users"], 1)
                self.assertEqual(set(meta["splits"]), {"train"})
                self.assertEqual(meta["splits"]["train"]["input_rows"], 1)
                self.assertEqual(meta["splits"]["train"]["rows"], 0 if discard else 1)
                self.assertEqual(meta["splits"]["train"]["last_batch_size"], 0 if discard else 1)
                self.assertEqual(pq.read_table(output / "train.parquet").num_rows, 0 if discard else 1)


class HFInputTest(unittest.TestCase):
    def test_default_input_is_requested_hf_configuration(self):
        args = prepare_data.parse_args(["--output_dir", "output"])
        self.assertIsNone(args.data_dir)
        self.assertEqual(args.hf_revision, "main")
        self.assertEqual(prepare_data.HF_REPO, "yufan/user_profile_dataset")
        self.assertEqual(prepare_data.HF_CONFIG, "User_Profile_Layer3_For_RL_V1")

    def test_hf_main_pins_revision_fetches_only_target_shards_and_preserves_provenance(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_paths = {}
            for split in ("train", "test"):
                name = f"{prepare_data.HF_CONFIG}/{split}-00000-of-00001.parquet"
                path = root / f"{split}.parquet"
                raw = {**monthly_row(split), "split": split, "history_start": "2026-10-04",
                       "future_end": "2026-11-08", "raw_history_targets": [], "statistics": {"active_days": 4}}
                pq.write_table(pa.Table.from_pylist([raw]), path)
                source_paths[name] = path
            api = Mock()
            api.dataset_info.return_value = SimpleNamespace(sha="fixed-commit")
            api.list_repo_tree.return_value = [
                SimpleNamespace(path=name) for name in [
                    *source_paths, f"{prepare_data.HF_CONFIG}/generation_info.json",
                    "Unrelated_Config/train-00000-of-00001.parquet",
                ]
            ]
            client = Mock(return_value=api)
            download = Mock(side_effect=lambda **kwargs: str(source_paths[kwargs["filename"]]))
            hub = SimpleNamespace(HfApi=client, hf_hub_download=download)
            output = root / "output"
            args = prepare_data.parse_args(["--output_dir", str(output), "--checkpoint", "synthetic",
                                           "--hf_revision", "requested-branch"])
            observed = {}

            def fake_workers(settings, locations, temporary):
                observed["source_paths"] = [location.source_file for location in locations]
                records = [prepare_data.load_location(location) for location in locations]
                fake = FakeInference()
                snapshots, rejected, _ = pipeline.build_profiles(records, fake.encode, fake.run)
                self.assertEqual(rejected, {})
                path = temporary / "worker-0.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    for location, record in zip(locations, records):
                        row, reason = pipeline.make_example(record, snapshots[record.user_id],
                                                           fake.encode, settings.max_prompt_length)
                        stream.write(json.dumps({"index": location.index, "split": location.split,
                                                 "user_id": record.user_id, "source_file": record.source_file,
                                                 "source_line": record.line, "reason": reason, "row": row}) + "\n")
                return [path]

            with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    patch.object(prepare_data, "hf_token", return_value="unit-test-token"), \
                    patch.object(prepare_data, "parse_args", return_value=args), \
                    patch.object(prepare_data, "run_workers", side_effect=fake_workers), patch("builtins.print"):
                prepare_data.main()
            client.assert_called_once_with(token="unit-test-token")
            api.dataset_info.assert_called_once_with(prepare_data.HF_REPO, revision="requested-branch")
            api.list_repo_tree.assert_called_once_with(
                prepare_data.HF_REPO, path_in_repo=prepare_data.HF_CONFIG, repo_type="dataset",
                revision="fixed-commit", recursive=False)
            self.assertEqual({call.kwargs["filename"] for call in download.call_args_list}, set(source_paths))
            self.assertTrue(all(call.kwargs["revision"] == "fixed-commit" for call in download.call_args_list))
            meta = json.loads((output / "meta.json").read_text())
            self.assertEqual(meta["source"]["revision"], "fixed-commit")
            self.assertEqual(meta["source"]["config"], prepare_data.HF_CONFIG)
            self.assertEqual(meta["splits"]["train"]["rows"], 1)
            self.assertEqual(meta["splits"]["test"]["rows"], 1)
            self.assertTrue(all(path.startswith("hf://") and "@fixed-commit/" in path
                                for path in observed["source_paths"]))
            self.assertEqual([item["path"] for item in meta["inputs"]], observed["source_paths"])
            train = pq.read_table(output / "train.parquet").to_pylist()[0]
            self.assertEqual(train["extra_info"]["source_file"], observed["source_paths"][0])
            self.assertNotIn("unit-test-token", (output / "meta.json").read_text())
            self.assertEqual({path.name for path in output.iterdir()},
                             {"train.parquet", "test.parquet", "pools.parquet", "discarded.jsonl", "meta.json"})

    def test_missing_hf_split_fails_before_download(self):
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="fixed-commit")
        api.list_repo_tree.return_value = [
            SimpleNamespace(path=f"{prepare_data.HF_CONFIG}/train-00000-of-00001.parquet")]
        download = Mock()
        hub = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=download)
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": hub}), \
                patch.object(prepare_data, "hf_token", return_value=None):
            with self.assertRaisesRegex(ValueError, "No test Parquet"):
                prepare_data.download_hf_inputs(Path(directory), "main")
        download.assert_not_called()

    def test_missing_parquet_cutoff_is_not_inferred(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = monthly_row()
            del raw["cutoff"]
            pq.write_table(pa.Table.from_pylist([raw]), root / "input.parquet")
            with pq.ParquetFile(root / "input.parquet") as parquet:
                with self.assertRaisesRegex(ValueError, "missing monthly input columns.*cutoff"):
                    prepare_data.write_parquet_jsonl(parquet, root / "train.jsonl")
            self.assertFalse((root / "train.jsonl").exists())

    def test_debug_hf_uses_range_reads_and_stops_at_total_limit_across_shards(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"{prepare_data.HF_CONFIG}/train-{i:05d}-of-00002.parquet" for i in range(2)]
            local = {}
            for shard, users in enumerate((("a", "b"), ("c", "d", "e"))):
                path = root / f"source-{shard}.parquet"
                pq.write_table(pa.Table.from_pylist([monthly_row(user) for user in users]), path)
                local[f"datasets/{prepare_data.HF_REPO}@fixed-commit/{names[shard]}"] = path
            test_name = f"{prepare_data.HF_CONFIG}/test-00000-of-00001.parquet"
            for limit, expected in ((1, ["a"]), (3, ["a", "b", "c"]), (9, ["a", "b", "c", "d", "e"])):
                with self.subTest(limit=limit):
                    api = Mock()
                    api.dataset_info.return_value = SimpleNamespace(sha="fixed-commit")
                    api.list_repo_tree.return_value = [SimpleNamespace(path=name) for name in [*names, test_name]]
                    remote = SimpleNamespace(open=Mock(side_effect=lambda path, *args, **kwargs: local[path].open("rb")))
                    download = Mock()
                    hub = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=download,
                                          HfFileSystem=Mock(return_value=remote))
                    with patch.dict(sys.modules, {"huggingface_hub": hub}), \
                            patch.object(prepare_data, "hf_token", return_value=None), patch("builtins.print"):
                        staged, sources, meta = prepare_data.download_hf_inputs(root / f"case-{limit}", "main", limit)
                    locations, _ = prepare_data.index_inputs(staged, 0, sources, max_users=limit)
                    self.assertEqual({item.user_id for item in locations}, set(expected))
                    self.assertTrue(all(item.split == "train" for item in locations))
                    self.assertEqual(remote.open.call_count, 1 if limit == 1 else 2)
                    self.assertTrue(all("/train-" in call.args[0] for call in remote.open.call_args_list))
                    download.assert_not_called()
                    self.assertEqual(meta["read_mode"], "range")
                    self.assertEqual(sum(item["rows"] for item in meta["files"]), len(expected))

    def test_parquet_debug_limit_across_record_batches(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([monthly_row(str(i)) for i in range(130)]),
                           root / "input.parquet", row_group_size=64)
            with pq.ParquetFile(root / "input.parquet") as parquet:
                count = prepare_data.write_parquet_jsonl(parquet, root / "train.jsonl", max_rows=65)
            rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            self.assertEqual(count, 65)
            self.assertEqual(len(rows), 65)
            self.assertEqual(rows[-1]["user_id"], "64")


if __name__ == "__main__":
    unittest.main()
