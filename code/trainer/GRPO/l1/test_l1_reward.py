"""CPU logic tests with deterministic embeddings and tokenizer counts; no pretrained models or GPUs."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
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
import rl_data
import utils


class Vector:
    def __init__(self, values):
        self.values = values

    def topk(self, k):
        return SimpleNamespace(indices=Vector(sorted(range(len(self.values)), key=lambda i: (-self.values[i], i))[:k]))

    def tolist(self):
        return list(self.values)


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

    def max(self, dim):
        if dim != 0:
            raise ValueError("Test matrix supports max only over rows")
        columns = [list(column) for column in zip(*self.rows)]
        return SimpleNamespace(values=Vector([max(column) for column in columns]),
                               indices=Vector([column.index(max(column)) for column in columns]))

    def tolist(self):
        return self.rows


class FakeEmbedder:
    """Past action P<n> matches I<n>'s queries, next-week action F<n> matches I<n>'s (n < 20); G<n> is another user's
    common action (a messenger, say) that every query slightly matches, and a topic named "common" strongly."""

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
                    head, _, tail = text.partition(":")
                    index = int(head[1:])
                    vector[index if index < 25 else 64] = 1.0
                    vector[32 + index if index < 20 else 65] = 1.0
                    tail = tail.strip()
                    vector[63] = 3.0 if tail == "common" else 1.0
                    if tail[:1] == "P" and tail[1:].isdigit():  # a topic named after a past action leans toward it
                        vector[int(tail[1:])] += 2.0
                elif text in ("PC", "FC"):
                    vector[0] = vector[32] = 4.0
                elif text in ("DP", "DF"):
                    vector[64 if text == "DP" else 65] = 1.0
                elif text[:1] == "G" and text[1:].isdigit():
                    vector[63] = 0.5
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


def topics_answer(interests):
    """One interest I<i> per item of `interests`, a list of (topic, evidence) pairs."""
    return json.dumps({"interests": [
        {"interest_name": f"I{index}", "actual_activity": "Synthetic activity",
         "topics": [{"topic": topic, "evidence": evidence} for topic, evidence in topics]}
        for index, topics in enumerate(interests)
    ]})


def fake_parquet(files):
    """A pyarrow.parquet stand-in reading `files` ({file name: rows}) by the name of the path."""
    class Table:
        def __init__(self, rows, columns):
            self.rows = [{column: row[column] for column in columns} for row in rows]

        def column(self, index):
            name = list(self.rows[0])[index] if self.rows else None
            return SimpleNamespace(to_pylist=lambda: [row[name] for row in self.rows])

        def take(self, numbers):
            table = Table([], [])
            table.rows = [self.rows[number] for number in numbers]
            return table

        def to_pylist(self):
            return list(self.rows)

    module = ModuleType("pyarrow.parquet")
    module.read_table = lambda path, columns: Table(files[Path(path).name], columns)
    return module


def fake_tokenizer(token_counts=None):
    counts = {} if token_counts is None else token_counts
    return Mock(encode=Mock(side_effect=lambda text, *, add_special_tokens: list(range(counts.get(text, 1)))))


def judge(prompt, text, finish_reason, tokenizer):
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
        self.scorer = reward.L1Reward(embedder=self.embedder, tokenizer=fake_tokenizer())
        self.past = [f"P{index}" for index in range(25)]
        self.future = [f"F{index}" for index in range(20)]
        self.others = [f"G{index}" for index in range(20)]  # other users' common next-week actions
        self.scorer.set_pools(self.past + ["DP"], self.future + ["DF"] + self.others)
        self.target = reward.PAST_COVERAGE_TARGET * 25

    def score(self, count=1, evidence=None):
        return self.scorer(content(self.past), answer(count, evidence), None, self.future)

    def test_two_signal_mean_without_interest_cap(self):
        for count in (1, 5, 9, 12, 18, 27):
            with self.subTest(count=count):
                result = self.score(count)
                # Every interest cites only P0, which I0 owns: 1 correct, below the coverage target of 25.
                self.assertAlmostEqual(result["past_assignment"], 1 / self.target)
                # I<n> retrieves F<n>; the rest of the top 20 are other users' actions.
                self.assertAlmostEqual(result["future_retrieval"], min(count, 20) / 20)
                self.assertAlmostEqual(result["reward"], (1 / self.target + min(count, 20) / 20) / 2)
                self.assertTrue(0 <= result["reward"] <= 1)

    def test_future_retrieval_picks_the_users_actions_out_of_a_fixed_budget(self):
        self.scorer.set_pools(self.past + ["DP"], ["F0", "F1", "G0", "G1"])

        def future(interests, targets=("F0", "F1")):
            return self.scorer(content(self.past), topics_answer(interests), None, list(targets))

        # B = 2: one topic finds F0, and an other user's action fills the second slot.
        self.assertEqual(future([[("topic", [0])]])["future_retrieval"], 0.5)
        both = future([[("topic", [0])], [("topic", [1])]])
        self.assertEqual(both["future_retrieval"], 1)
        self.assertEqual(both["hits"], [[[0, 1]], [[1, 2]]])
        self.assertEqual(both["queries"], [["I0: topic"], ["I1: topic"]])
        # A sibling topic retrieving the same actions adds nothing.
        self.assertEqual(future([[("topic", [0]), ("other", [0])]])["future_retrieval"], 0.5)
        # A topic common to every user ranks their actions too and pushes this user's out of the budget.
        common = future([[("topic", [0])], [("topic", [1])], [("common", [2])]])
        self.assertEqual(common["future_retrieval"], 0)
        # B counts distinct next-week actions; a target missing from the pool is still ranked.
        self.assertEqual(future([[("topic", [0])], [("topic", [1])]], ("F0", "F0", "F1"))["future_retrieval"], 1)
        self.scorer.set_pools(self.past + ["DP"], ["F0", "G0", "G1"])
        self.assertEqual(future([[("topic", [0])], [("topic", [1])]])["future_retrieval"], 1)

    def test_past_assignment_credits_actions_owned_by_their_citing_topic(self):
        def past(interests):
            return self.scorer(content(self.past), topics_answer(interests), None, self.future)

        owned = past([[("topic", [index])] for index in range(5)])
        self.assertAlmostEqual(owned["past_assignment"], 5 / self.target)
        self.assertEqual(owned["past_owners"][:6], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 0]])
        swapped = past([[("topic", [1])], [("topic", [0])]])
        self.assertEqual(swapped["past_assignment"], 0)
        # Every topic competes and earns; 20 correct citations pass the target of 17.5.
        every = past([[("topic", [index])] for index in range(20)])
        self.assertEqual(every["past_assignment"], 1)
        # One topic citing everything owns everything: the extreme future_retrieval must balance.
        bucket = past([[("topic", list(range(25)))]])
        self.assertEqual(bucket["past_assignment"], 1)
        self.assertEqual(bucket["future_retrieval"], 1 / 20)

    def test_past_assignment_coverage_target(self):
        def past(interests):
            return self.scorer(content(self.past), topics_answer(interests), None, self.future)

        self.assertEqual(self.target, 17.5)
        # Below the target, fewer citations cost even when all are correct.
        self.assertAlmostEqual(past([[("topic", [index])] for index in range(10)])["past_assignment"], 10 / 17.5)
        # Past the target, skipping is free: 18 or 20 correct citations both score 1.
        self.assertEqual(past([[("topic", [index])] for index in range(18)])["past_assignment"], 1)
        # Citing the rest wrongly (I20.. each cites the action the next one owns) costs: 20 correct of 25 cited.
        shifted = [[("topic", [index])] for index in range(20)] + [[("topic", [20 + (index + 1) % 5])]
                                                                    for index in range(5)]
        self.assertAlmostEqual(past(shifted)["past_assignment"], 20 / 25)

    def test_action_weight(self):
        self.assertEqual(reward.action_weight(None), 1)
        self.assertEqual(reward.action_weight(0), 1)
        self.assertEqual(reward.action_weight(reward.ACTION_FREQ_HALF_WEIGHT), 0.5)
        self.assertAlmostEqual(reward.action_weight(110703), 1 / (1 + 110703 / 100))

    def test_ubiquitous_actions_earn_nothing_in_either_signal(self):
        hot, w = 10000, reward.action_weight(10000)
        past_freqs = [hot] * 5 + [0] * 20  # P0..P4 are ubiquitous

        def past(cited):
            # I<n> owns P<n>; interests that cite nothing still exist (they own their action, uncited).
            interests = [[("topic", [index] if index in cited else [])] for index in range(max(cited) + 1)]
            return self.scorer(content(self.past), topics_answer(interests), None, self.future, past_freqs, None)

        # 18 correct citations pass the unweighted target either way, but ubiquitous ones fill no coverage now.
        with_hot = past(set(range(18)))
        target = 0.7 * (5 * w + 20)
        self.assertEqual(with_hot["past_assignment_unweighted"], 1)
        self.assertAlmostEqual(with_hot["past_assignment"], (5 * w + 13) / target)
        tail_only = past(set(range(5, 23)))
        self.assertEqual(tail_only["past_assignment"], 1)
        self.assertGreater(tail_only["past_assignment"], with_hot["past_assignment"])
        # Future: I0..I4 retrieve F0..F4 and other users' actions fill the rest of the top 20.
        hot_future = [hot] * 5 + [0] * 15
        retrieved = self.scorer(content(self.past), answer(5), None, self.future, None, hot_future)
        self.assertAlmostEqual(retrieved["future_retrieval_unweighted"], 5 / 20)
        self.assertAlmostEqual(retrieved["future_retrieval"], 5 * w / (5 * w + 15))
        tail_future = self.scorer(content(self.past), answer(5), None, self.future, None, [0] * 5 + [hot] * 15)
        self.assertAlmostEqual(tail_future["future_retrieval"], 5 / (5 + 15 * w))
        # Without frequencies every action weighs 1, and the scale of the weights does not matter.
        plain = self.scorer(content(self.past), answer(5), None, self.future)
        self.assertEqual(plain["past_assignment"], plain["past_assignment_unweighted"])
        self.assertEqual(plain["future_retrieval"], plain["future_retrieval_unweighted"])
        uniform = self.scorer(content(self.past), answer(5), None, self.future, [500] * 25, [500] * 20)
        self.assertAlmostEqual(uniform["past_assignment"], plain["past_assignment"])
        self.assertAlmostEqual(uniform["future_retrieval"], plain["future_retrieval"])

    def test_duplicate_future_targets_keep_their_first_frequency(self):
        result = self.scorer(content(self.past), answer(1), None, ["F0", "F0", "F1"], None, [0, 10000, 10000])
        self.assertAlmostEqual(result["future_retrieval"], 1 / (1 + reward.action_weight(10000)))

    def test_frequency_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "past_freqs has 1 values for 25 input signals"):
            self.scorer(content(self.past), answer(1), None, self.future, [1], None)
        with self.assertRaisesRegex(ValueError, "future_freqs has 1 values for 20 next-week actions"):
            self.scorer(content(self.past), answer(1), None, self.future, None, [1])

    def test_past_assignment_is_exact_to_the_topic(self):
        def past(topics):
            return self.scorer(content(self.past), topics_answer([topics]), None, self.future)

        exact = past([("P1", [1]), ("P2", [2])])
        self.assertAlmostEqual(exact["past_assignment"], 2 / self.target)
        self.assertEqual(exact["past_owners"][1:3], [[0, 0], [0, 1]])
        # The same interest is not enough: each action must be cited by the topic that owns it.
        self.assertEqual(past([("P1", [2]), ("P2", [1])])["past_assignment"], 0)
        # A tie (two topics with the same query) counts when a citing topic is among the tied.
        tied = past([("a", []), ("b", [0])])
        self.assertAlmostEqual(tied["past_assignment"], 1 / self.target)
        self.assertEqual(tied["past_owners"][0], [0, 0])

    def test_coverage_is_diagnostic_and_future_ignores_evidence(self):
        sparse = self.score(evidence=[0])
        dense = self.score(evidence=list(range(25)))
        self.assertEqual(sparse["coverage"], 1 / 25)
        self.assertEqual(dense["coverage"], 1)
        self.assertAlmostEqual(sparse["past_assignment"], 1 / self.target)
        self.assertEqual(dense["past_assignment"], 1)
        self.assertEqual(sparse["future_retrieval"], dense["future_retrieval"])

    def test_past_never_ranks_other_users_actions(self):
        original = self.score()
        self.scorer.set_pools(self.past + ["DP", "PC"], self.future + ["DF"] + self.others)
        changed_past = self.score()
        self.assertEqual(changed_past["past_assignment"], original["past_assignment"])
        self.assertEqual(changed_past["future_retrieval"], original["future_retrieval"])
        self.scorer.set_pools(self.past + ["DP"], self.future + ["DF", "FC"] + self.others)
        self.assertEqual(self.score()["past_assignment"], original["past_assignment"])

    def test_query_and_shared_action_embeddings_are_reused(self):
        before = self.embedder.clear_count
        self.scorer.set_pools(["P0", "P0"], ["P0", "F0", "F0"])
        self.assertEqual(self.embedder.clear_count, before + 1)
        self.assertEqual(self.scorer.past_pool, ["P0"])
        self.assertEqual(self.scorer.future_pool, ["P0", "F0"])
        self.assertEqual(self.embedder.computed.count((False, "P0")), 1)
        self.scorer(content(["P0"]), answer(1), None, ["F0"])
        # One query encoding serves both signals, and the user's actions come from the pool.
        self.assertEqual(sum(query for query, _ in self.embedder.calls), 1)
        self.assertEqual(self.embedder.computed.count((False, "P0")), 1)

    def test_invalid_and_truncated_answers(self):
        invalid = self.scorer(content(self.past), "not JSON", None, self.future)
        self.assertEqual(invalid["reward"], 0)
        self.assertTrue(all(invalid[name] == 0 for name in reward.SIGNALS))
        truncated = self.scorer(content(self.past), answer(1), "length", self.future)
        self.assertEqual(truncated["reward"], 0)
        self.assertGreater(truncated["past_assignment"], 0)
        self.assertGreater(truncated["future_retrieval"], 0)

    def test_empty_answer_and_future_targets(self):
        empty = self.score(0)
        self.assertEqual(empty["reward"], 0)
        self.assertEqual(empty["past_assignment"], 0)
        self.assertEqual(empty["future_retrieval"], 0)
        self.assertIsNone(empty["past_owners"])
        no_future = self.scorer(content(self.past), answer(1), None, [])
        self.assertEqual(no_future["future_retrieval"], 0)
        self.assertAlmostEqual(no_future["reward"], (1 / self.target) / 2)

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
                                                  {"batch_id": "train-000000", "content": content(self.past),
                                                   "past_freq": [10000] + [1] * 24, "future_freq": [1] * 20},
                                                  "pools.parquet", tokenizer=self.scorer.tokenizer)
            loader.assert_called_once_with("pools.parquet", "train-000000")
        self.assertEqual(set(result), {"score", "json_valid", "rule_pass", "n_interests", *reward.SIGNALS,
                                       *reward.STRUCTURE_SIGNALS})
        self.assertTrue(all(isinstance(value, float) for value in result.values()))
        # The frequencies reach the reward: I0 cites and owns only P0, which is ubiquitous.
        self.assertAlmostEqual(result["past_assignment_unweighted"], 1 / self.target)
        self.assertLess(result["past_assignment"], 0.001)
        with patch.object(verl_reward, "_state", {"reward_fn": self.scorer, "batch_id": "train-000000"}):
            invalid = verl_reward.compute_score("user_profile_l1", "not JSON", self.future,
                                                {"batch_id": "train-000000", "content": content(self.past),
                                                 "past_freq": [1] * 25, "future_freq": [1] * 20},
                                                "pools.parquet", tokenizer=self.scorer.tokenizer)
        self.assertTrue(all(invalid[name] is None for name in reward.STRUCTURE_SIGNALS))

    def test_structure_signals_average_within_the_answer(self):
        self.scorer.tokenizer = fake_tokenizer({"I0": 2, "I1": 4, "a": 1, "b": 2, "c": 3})
        output = {"interests": [
            {"interest_name": "I0", "actual_activity": "Synthetic activity",
             "topics": [{"topic": "a", "evidence": [0, 1, 1]}, {"topic": "b", "evidence": [2]}]},
            {"interest_name": "I1", "actual_activity": "Synthetic activity",
             "topics": [{"topic": "c", "evidence": [3, 4, 5]}]},
        ]}
        result = self.scorer(content(self.past), json.dumps(output), None, self.future)
        self.assertEqual(result["topics_per_interest"], 1.5)
        self.assertEqual(result["evidence_per_topic"], 2)  # distinct idx: 2, 1, 3
        self.assertEqual(result["interest_name_tokens"], 3)
        self.assertEqual(result["topic_tokens"], 2)
        for text in ("not JSON", answer(0)):
            with self.subTest(text=text):
                empty = self.scorer(content(self.past), text, None, self.future)
                self.assertTrue(all(empty[name] is None for name in reward.STRUCTURE_SIGNALS))


class TokenLimitRewardTest(unittest.TestCase):
    def setUp(self):
        language = patch.object(reward.user_profile_rules, "language_tools")
        language_check = patch.object(reward.user_profile_rules, "check_language")
        language.start()
        language_check.start()
        self.addCleanup(language.stop)
        self.addCleanup(language_check.stop)
        self.detect = patch.object(reward.user_profile_rules, "detect_language", return_value="en").start()
        self.addCleanup(patch.stopall)
        self.scorer = reward.L1Reward(embedder=FakeEmbedder(), tokenizer=fake_tokenizer())
        self.scorer.set_pools(["P0", "DP"], ["F0", "DF"])

    def score(self, output=None, finish_reason=None):
        text = answer(1) if output is None else json.dumps(output)
        return self.scorer(content(["P0"]), text, finish_reason, ["F0"])

    def test_non_english_or_untagged_interests_allow_ten_tokens(self):
        for language in ("zh-Hans", "es", "mix", None):
            for label, violation in (("I0", "interest_name_too_long"), ("topic", "topic_too_long")):
                for count in (7, 10, 11):
                    with self.subTest(language=language, label=label, tokens=count):
                        self.detect.return_value = language
                        self.scorer.tokenizer = fake_tokenizer({label: count})
                        result = self.score()
                        self.assertEqual(result["rule_pass"], count <= 10)
                        self.assertEqual(result["violations"], [] if count <= 10 else [violation])

    def test_limit_is_per_interest_by_its_cited_actions(self):
        self.detect.side_effect = lambda actions: "en" if actions == ["P0"] else "ja"
        output = json.loads(topics_answer([[("english", [0])], [("japanese", [1])]]))
        text = json.dumps(output)
        self.scorer.tokenizer = fake_tokenizer({"english": 6, "japanese": 10})
        self.assertTrue(self.scorer(content(["P0", "DP"]), text, None, ["F0"])["rule_pass"])
        self.scorer.tokenizer = fake_tokenizer({"english": 7, "japanese": 10})
        result = self.scorer(content(["P0", "DP"]), text, None, ["F0"])
        self.assertEqual(result["violations"], ["topic_too_long"])
        self.detect.assert_any_call(["P0"])
        self.detect.assert_any_call(["DP"])

    def test_six_tokens_pass_and_seven_or_more_fail_for_each_field(self):
        for label, violation in (("I0", "interest_name_too_long"), ("topic", "topic_too_long")):
            for count in (1, 6, 7, 8):
                with self.subTest(label=label, tokens=count):
                    tokenizer = fake_tokenizer({label: count})
                    self.scorer.tokenizer = tokenizer
                    result = self.score()
                    self.assertTrue(result["json_valid"])
                    self.assertEqual(result["rule_pass"], count < 7)
                    self.assertEqual(result["reward"], 1.0 if count < 7 else 0.0)
                    self.assertEqual(result["violations"], [] if count < 7 else [violation])
                    self.assertEqual(result["past_assignment"], 1.0)
                    self.assertEqual(result["future_retrieval"], 1.0)
                    tokenizer.encode.assert_any_call(label, add_special_tokens=False)
                    self.assertTrue(all(call.kwargs == {"add_special_tokens": False}
                                        for call in tokenizer.encode.call_args_list))

    def test_name_of_a_later_interest_rejects_whole_rollout(self):
        self.scorer.tokenizer = fake_tokenizer({"I1": 7})
        result = self.score(json.loads(answer(2)))
        self.assertFalse(result["rule_pass"])
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["violations"], ["interest_name_too_long"])
        self.assertEqual(result["past_assignment"], 1.0)

    def test_second_topic_rejects_whole_rollout(self):
        output = json.loads(answer(1))
        output["interests"][0]["topics"].append({"topic": "second", "evidence": [0]})
        self.scorer.tokenizer = fake_tokenizer({"second": 7})
        result = self.score(output)
        self.assertEqual(result["queries"], [["I0: topic", "I0: second"]])
        self.assertFalse(result["rule_pass"])
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["violations"], ["topic_too_long"])

    def test_topic_of_a_later_interest_rejects_whole_rollout(self):
        output = json.loads(answer(2))
        output["interests"][1]["topics"][0]["topic"] = "uncounted"
        self.scorer.tokenizer = fake_tokenizer({"uncounted": 7})
        result = self.score(output)
        self.assertFalse(result["rule_pass"])
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["violations"], ["topic_too_long"])

    def test_actual_activity_is_not_token_limited(self):
        tokenizer = fake_tokenizer({"I0": 6, "topic": 6, "Synthetic activity": 100})
        self.scorer.tokenizer = tokenizer
        result = self.score()
        self.assertTrue(result["rule_pass"])
        self.assertEqual(result["reward"], 1.0)
        self.assertNotIn("Synthetic activity", [call.args[0] for call in tokenizer.encode.call_args_list])

    def test_new_violations_preserve_existing_evidence_and_truncation_failures(self):
        self.scorer.tokenizer = fake_tokenizer({"I0": 7, "topic": 7})
        output = json.loads(answer(1, evidence=[99]))
        result = self.score(output, finish_reason="length")
        self.assertTrue(result["json_valid"])
        self.assertFalse(result["rule_pass"])
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(set(result["violations"]),
                         {"interest_name_too_long", "topic_too_long", "unknown_evidence_idx", "truncated"})

    def test_malformed_output_is_not_tokenized(self):
        for text in ("not JSON", "{}"):
            with self.subTest(text=text):
                result = self.scorer(content(["P0"]), text, None, ["F0"])
                self.assertFalse(result["json_valid"])
                self.assertFalse(result["rule_pass"])
                self.assertEqual(result["reward"], 0.0)
        self.scorer.tokenizer.encode.assert_not_called()

    def test_tokenizer_errors_are_not_silently_ignored(self):
        self.scorer.tokenizer.encode.side_effect = ValueError("tokenizer failed")
        with self.assertRaisesRegex(ValueError, "tokenizer failed"):
            self.score()

    def test_adapter_returns_zero_for_overlong_label(self):
        self.scorer.tokenizer = fake_tokenizer({"I0": 7})
        pools = {"past_actions": ["P0", "DP"], "future_actions": ["F0", "DF"]}
        with patch.object(verl_reward, "_state", {"reward_fn": self.scorer, "batch_id": None}), \
                patch.object(verl_reward, "load_pools", return_value=pools):
            result = verl_reward.compute_score("user_profile_l1", answer(1), ["F0"],
                                              {"batch_id": "test", "content": content(["P0"]),
                                               "past_freq": [1], "future_freq": [1]},
                                              "pools.parquet", tokenizer=self.scorer.tokenizer)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["rule_pass"], 0.0)
        self.assertEqual(result["json_valid"], 1.0)

    def test_worker_copies_local_tokenizer_once_without_loading_a_checkpoint(self):
        tokenizer = fake_tokenizer({"label": 7})
        modules = {"torch": SimpleNamespace(set_num_threads=Mock()), "transformers": None}
        with patch.dict(sys.modules, modules), \
                patch.object(verl_reward, "_state", {"reward_fn": None, "batch_id": None}), \
                patch.object(verl_reward, "L1Reward", return_value=self.scorer) as constructor:
            for _ in range(2):
                self.assertIs(verl_reward.reward_fn("embedding-model", "http://embedding", 256,
                                                   tokenizer), self.scorer)
        private_tokenizer = constructor.call_args.kwargs["tokenizer"]
        constructor.assert_called_once_with("embedding-model", 256, embedding_url="http://embedding",
                                            tokenizer=private_tokenizer)
        self.assertIsNot(private_tokenizer, tokenizer)
        self.assertEqual(len(private_tokenizer.encode("label", add_special_tokens=False)), 7)
        tokenizer.encode.assert_not_called()

    def test_concurrent_answers_initialize_scorer_and_batch_only_once(self):
        pools = {"past_actions": ["P0", "DP"], "future_actions": ["F0", "DF"]}
        tokenizer = fake_tokenizer()

        def score(_):
            return verl_reward.compute_score("user_profile_l1", answer(1), ["F0"],
                                             {"batch_id": "test", "content": content(["P0"]),
                                              "past_freq": [1], "future_freq": [1]},
                                             "pools.parquet", tokenizer=tokenizer)

        with patch.dict(sys.modules, {"torch": SimpleNamespace(set_num_threads=Mock()), "transformers": None}), \
                patch.object(verl_reward, "_state", {"reward_fn": None, "batch_id": None}), \
                patch.object(verl_reward, "L1Reward", return_value=self.scorer) as constructor, \
                patch.object(verl_reward, "load_pools", return_value=pools) as loader:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(score, range(16)))
        constructor.assert_called_once()
        loader.assert_called_once_with("pools.parquet", "test")
        self.assertTrue(all(result["score"] == 1.0 and result["rule_pass"] == 1.0 for result in results))


class RewardLoaderTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.importer = Mock()
        modules = {
            "verl": SimpleNamespace(DataProto=object),
            "verl.utils.reward_score": SimpleNamespace(get_default_compute_score=Mock()),
            "verl.utils.import_utils": SimpleNamespace(load_extern_object=self.importer),
        }
        dependencies = patch.dict(sys.modules, modules)
        dependencies.start()
        self.addCleanup(dependencies.stop)
        path = Path(reward.__file__).resolve().parents[1] / "verl" / "trainer" / "ppo" / "reward.py"
        spec = importlib.util.spec_from_file_location("l1_test_reward_loader", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load reward loader from {path}")
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)

    def config(self, **kwargs):
        return SimpleNamespace(reward={"custom_reward_function": {
            "path": "synthetic.py", "name": "compute_score", "reward_kwargs": kwargs,
        }})

    def test_callbacks_bind_each_workers_own_tokenizer_without_changing_config(self):
        def callback(text, *, tokenizer, multiplier):
            return tokenizer, len(tokenizer.encode(text, add_special_tokens=False)) * multiplier

        self.importer.return_value = callback
        config = self.config(multiplier=2)
        first, second = fake_tokenizer({"label": 6}), fake_tokenizer({"label": 7})
        first_fn = self.loader.get_custom_reward_fn(config, tokenizer=first)
        second_fn = self.loader.get_custom_reward_fn(config, tokenizer=second)
        self.assertEqual(first_fn("label"), (first, 12))
        self.assertEqual(second_fn("label"), (second, 14))
        self.assertEqual(first_fn("label"), (first, 12))
        self.assertEqual(config.reward["custom_reward_function"]["reward_kwargs"], {"multiplier": 2})

    def test_callback_without_tokenizer_parameter_keeps_its_signature(self):
        def callback(text, *, multiplier):
            return len(text) * multiplier

        self.importer.return_value = callback
        function = self.loader.get_custom_reward_fn(self.config(multiplier=2), tokenizer=fake_tokenizer())
        self.assertEqual(function("abc"), 6)

    def test_kwargs_only_callback_does_not_opt_in_to_tokenizer_injection(self):
        def callback(**kwargs):
            return kwargs

        self.importer.return_value = callback
        function = self.loader.get_custom_reward_fn(self.config(multiplier=2), tokenizer=fake_tokenizer())
        self.assertEqual(function(text="abc"), {"text": "abc", "multiplier": 2})

    def test_async_callback_receives_local_tokenizer(self):
        async def callback(text, tokenizer):
            return tokenizer, len(tokenizer.encode(text, add_special_tokens=False))

        self.importer.return_value = callback
        tokenizer = fake_tokenizer({"label": 7})
        function = self.loader.get_custom_reward_fn(self.config(), tokenizer=tokenizer)
        self.assertEqual(asyncio.run(function("label")), (tokenizer, 7))

    def test_reward_manager_injects_its_generation_tokenizer(self):
        def callback(text, tokenizer):
            return tokenizer, len(tokenizer.encode(text, add_special_tokens=False))

        self.importer.return_value = callback
        config = self.config()
        config.reward = SimpleNamespace(get=config.reward.get, reward_manager=SimpleNamespace(name="naive"))
        tokenizer = fake_tokenizer({"label": 7})
        with patch.object(self.loader, "resolve_reward_manager_cls", return_value=SimpleNamespace):
            manager = self.loader.load_reward_manager(config, tokenizer)
        self.assertIs(manager.tokenizer, tokenizer)
        self.assertEqual(manager.compute_score("label"), (tokenizer, 7))

    def test_legacy_loader_without_tokenizer_still_works(self):
        self.importer.return_value = lambda text: len(text)
        function = self.loader.get_custom_reward_fn(self.config())
        self.assertEqual(function("abc"), 3)

    def test_no_custom_reward_is_unchanged(self):
        config = SimpleNamespace(reward={})
        self.assertIsNone(self.loader.get_custom_reward_fn(config, tokenizer=fake_tokenizer()))
        self.importer.assert_not_called()


class PoolPreparationTest(unittest.TestCase):
    def launch_with_meta(self, meta: dict) -> subprocess.CompletedProcess:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "train.parquet").touch()
            (data / "meta.json").write_text(json.dumps(meta))
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "python3").symlink_to(sys.executable)
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "DATA_DIR": str(data), "OUTPUT_DIR": str(root / "output"), "LOGGER": "['console']",
                   "TRAIN_BATCH_SIZE": "64", "MAX_PROMPT_LENGTH": "17408", "MAX_RESPONSE_LENGTH": "8192",
                   "SFT_REPLAY": "1", "SFT_MAX_SEQ_LENGTH": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            launcher = Path(prepare_data.__file__).resolve().parents[1] / "run_l1_grpo.sh"
            return subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=30)

    def test_launcher_rejects_old_cache_before_other_preparation(self):
        old_dataset = self.launch_with_meta({"batch_size": 64, "max_prompt_length": 17408})
        self.assertNotEqual(old_dataset.returncode, 0)
        self.assertIn("RL_User_Profile_Layer1_V1", old_dataset.stderr)
        self.assertIn("regenerate", old_dataset.stderr)
        old_pools = self.launch_with_meta({"dataset": prepare_data.DATASET, "batch_size": 64,
                                           "max_prompt_length": 17408})
        self.assertNotEqual(old_pools.returncode, 0)
        self.assertIn("expected single-language dual past/future pools", old_pools.stderr)
        self.assertIn("regenerate", old_pools.stderr)

    def test_rows_are_read_in_the_seeded_user_order(self):
        def behaviors(actions, date):
            return [{"date": date, "source": "Bing", "action": action, "action_idx": 0, "action_freq": 1}
                    for action in actions]

        files = {
            "train-00000-of-00002.parquet": [
                {"user_id": "u0", "language": "en", "past_behaviors": behaviors(["P0"], "2026-07-10"),
                 "future_behaviors": behaviors(["F0"], "2026-07-17")},
                {"user_id": "u1", "language": "en", "past_behaviors": [], "future_behaviors": behaviors(["F1"], "2026-07-17")},
            ],
            "train-00001-of-00002.parquet": [
                {"user_id": "u2", "language": "zh", "past_behaviors": behaviors(["P2"], "2026-07-10"),
                 "future_behaviors": behaviors(["F2"], "2026-07-17")},
                {"user_id": "u3", "language": "en", "past_behaviors": behaviors(["P3"], "2026-07-10"),
                 "future_behaviors": behaviors(["F3"], "2026-07-17")},
            ],
        }
        pq = fake_parquet(files)
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"pyarrow": ModuleType("pyarrow"),
                                                                         "pyarrow.parquet": pq}):
            root = Path(directory)
            for name in files:
                (root / name).touch()
            with prepare_data.rl_data_files(root, "train") as paths:
                order = prepare_data.index_rows(paths, 0)
                users = [files[Path(path).name][number]["user_id"] for path, number in order]
                self.assertEqual(users, sorted(["u0", "u1", "u2", "u3"],
                                               key=lambda user: rl_data.row_key(0, user.encode())))
                self.assertEqual([row["user_id"] for row in prepare_data.read_rows(paths[1], [1, 0])], ["u3", "u2"])
                args = argparse.Namespace(seed=0, tokenizer="synthetic", max_prompt_length=10000, num_workers=1,
                                          log_every=1)
                with patch.object(prepare_data.layer1, "fit_prompt", side_effect=lambda signals, encode: (signals, [1])), \
                        patch("builtins.print"):
                    rows, skipped = prepare_data.accepted(paths, args, "train", None, distinct=False,
                                                          encode=lambda text: [1])
            self.assertEqual(skipped, 1)  # u1 has no past signal
            self.assertEqual([row["extra_info"]["user_id"] for row in rows], [user for user in users if user != "u1"])
            first = rows[0]["extra_info"]
            self.assertEqual(files[first["file"]][first["row"]]["user_id"], first["user_id"])
            self.assertEqual({row["extra_info"]["user_id"]: row["extra_info"]["language"] for row in rows},
                             {"u0": "en", "u2": "zh", "u3": "en"})
            duplicate = {**files, "train-00001-of-00002.parquet": files["train-00000-of-00002.parquet"]}
            with patch.dict(sys.modules, {"pyarrow.parquet": fake_parquet(duplicate)}), \
                    self.assertRaisesRegex(SystemExit, "one row per user_id"):
                prepare_data.index_rows([root / name for name in files], 0)

    def test_past_pool_uses_only_visible_signals(self):
        past = [{"date": "2026-10-03", "source": "Bing", "action": action, "action_freq": freq}
                for action, freq in (("P0", 7), ("P1", 9))]
        row = {"user_id": "synthetic-user", "past_behaviors": past,
               "future_behaviors": [{"date": "2026-10-10", "source": "Bing", "action": "F0", "action_freq": 3}]}
        with patch.object(prepare_data.layer1, "fit_prompt", side_effect=lambda signals, encode: (signals[:1], [1, 2])):
            example = prepare_data.build_example(row, lambda text: [1], 10000)
        if example is None:
            raise AssertionError("The synthetic row should survive prompt preparation")
        self.assertEqual(example["extra_info"]["language"], "und")
        # The reward's action weights: one frequency per visible signal (prompt order) and per ground-truth action.
        self.assertEqual(example["extra_info"]["past_freq"], [7])
        self.assertEqual(example["extra_info"]["future_freq"], [3])
        self.assertNotIn("action_freq", example["extra_info"]["content"])
        pool = prepare_data.assign_batches([example], "train", 1)["train-000000"]
        self.assertEqual(pool, {"past_actions": ["P0"], "future_actions": ["F0"], "users": 1})
        stale = {**row, "future_behaviors": [{"date": "2026-10-10", "source": "Bing", "action": "F0"}]}
        with patch.object(prepare_data.layer1, "fit_prompt", side_effect=lambda signals, encode: (signals, [1])), \
                self.assertRaisesRegex(ValueError, "no action_freq"):
            prepare_data.build_example(stale, lambda text: [1], 10000)

    def test_clean_behaviors_keeps_the_action_frequency(self):
        long_action = "x" * 200
        behaviors = [{"date": "2026-10-03", "source": "Bing", "action": "A", "action_freq": 5},
                     {"date": "2026-10-04", "source": "Bing", "action": "A", "action_freq": 5},
                     {"date": "2026-10-03", "source": "Bing", "action": long_action, "action_freq": 2},
                     {"date": "2026-10-03", "source": "Bing", "action": long_action + "y", "action_freq": 9},
                     {"date": "2026-10-03", "source": "Bing", "action": "B"}]
        cleaned = {b["action"]: b["action_freq"] for b in rl_data.clean_behaviors("u", behaviors)}
        # Raw actions truncated to the same text share the highest frequency; a missing one stays None.
        self.assertEqual(cleaned, {"A": 5, "x" * 128: 9, "B": None})

    def test_batches_and_validation_pools(self):
        def row(past, future, language="en"):
            return {"extra_info": {"content": content(past), "language": language},
                    "reward_model": {"ground_truth": future}}
        rows = [row(["P0"], ["F0"]), row(["P1", "P0"], ["F1", "F0"])]
        pools = prepare_data.assign_batches(rows, "train", 1)
        self.assertEqual(pools["train-000000"], {"past_actions": ["P0"], "future_actions": ["F0"], "users": 1})
        # Validation: one pool per language, topped up with same-language distractors' future actions only.
        val = [row(["P2"], ["F2"], "sv"), row(["P0"], ["F0"]), row(["P1"], ["F1"])]
        distractors = [row(["P3"], ["F3"], "sv"), row(["P4"], ["F4"]), row(["P5"], ["F5"]), row(["P6"], ["F6"], "ja")]
        pools = prepare_data.assign_batches(val, "val", None, distractors, pool_users=3)
        self.assertEqual([r["extra_info"]["batch_id"] for r in val], ["val-en", "val-en", "val-sv"])
        self.assertEqual(pools["val-en"], {"past_actions": ["P0", "P1"], "future_actions": ["F0", "F1", "F4"],
                                           "users": 3})
        self.assertEqual(pools["val-sv"]["future_actions"], ["F2", "F3"])
        self.assertEqual(pools["val-sv"]["past_actions"], ["P2"])
        self.assertNotIn("val-ja", pools)

    def test_single_language_batches_in_a_seeded_order(self):
        rows = [{"extra_info": {"language": language, "user_id": f"{language}{index}"}}
                for index in range(5) for language in ("en", "zh", "sv")]
        batches = prepare_data.language_batches(rows, 2, 0)
        self.assertEqual(len(batches), 6)  # 2 full batches per language; each fifth row is dropped
        self.assertTrue(all(len({r["extra_info"]["language"] for r in batch}) == 1 and len(batch) == 2
                            for batch in batches))
        self.assertEqual(batches, prepare_data.language_batches(rows, 2, 0))
        self.assertNotEqual([b[0]["extra_info"]["user_id"] for b in batches],
                            [b[0]["extra_info"]["user_id"] for b in prepare_data.language_batches(rows, 2, 1)])
        self.assertEqual(sorted(r["extra_info"]["user_id"] for batch in batches for r in batch),
                         sorted(f"{language}{index}" for index in range(4) for language in ("en", "zh", "sv")))

    def test_manifest_and_parquet_writer_columns(self):
        pa, pq = ModuleType("pyarrow"), ModuleType("pyarrow.parquet")
        pa.parquet = pq
        pa.Table = SimpleNamespace(from_pylist=lambda rows: rows, from_pydict=lambda values: values)
        pq.write_table = Mock()
        def row(past, future):
            return {"extra_info": {"content": content(past), "prompt_tokens": 2, "language": "en"},
                    "reward_model": {"ground_truth": future}}
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.parquet": pq}):
            args = argparse.Namespace(output_dir=Path(directory), batch_size=1, seed=0,
                                      max_prompt_length=10000, tokenizer="synthetic")
            with patch("builtins.print"):
                prepare_data.write(args, [row(["P0"], ["F0"])], [row(["P1"], ["F1"])])
            meta = json.loads((Path(directory) / "meta.json").read_text())
        self.assertEqual(meta["pool_format_version"], prepare_data.POOL_FORMAT_VERSION)
        table = pq.write_table.call_args_list[2].args[0]
        self.assertEqual(table, {"batch_id": ["val-en", "train-000000"],
                                 "past_actions": [["P0"], ["P1"]], "future_actions": [["F0"], ["F1"]]})
        self.assertEqual(pq.write_table.call_args_list[2].kwargs, {"row_group_size": 1})


class CaseStudyTest(unittest.TestCase):
    def test_score_requires_generation_checkpoint(self):
        from case_study import layer1_sampling

        argv = ["layer1_sampling.py", "score", "--output_dir", "synthetic"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit) as error:
                layer1_sampling.parse_args()
        self.assertEqual(error.exception.code, 2)
        with patch.object(sys, "argv", [*argv, "--checkpoint", "generation-checkpoint"]):
            self.assertEqual(layer1_sampling.parse_args().checkpoint, "generation-checkpoint")

    def test_score_reports_both_sides(self):
        from case_study import layer1_sampling

        with TemporaryDirectory() as directory:
            root = Path(directory)
            signals = [{"date": "2026-10-03", "source": "Bing", "action": "P0", "action_freq": 1}]
            row = {"prompt_id": 0, "user_id": "synthetic", "content": content(["P0"]), "signals": signals,
                   "future": [{"date": "2026-10-10", "source": "Bing", "action": "F0", "action_freq": 1}],
                   "prompt_tokens": 2, "signals_dropped": 0,
                   "samples": [{"text": answer(1), "finish_reason": "stop"},
                               {"text": "not JSON", "finish_reason": "stop"}]}
            (root / "rollouts.jsonl").write_text(json.dumps(row) + "\n")
            args = argparse.Namespace(output_dir=root, checkpoint="generation-checkpoint",
                                      embedding_model="synthetic", embedding_batch_size=256,
                                      pool_users=64, embedding_url=None, report_prompts=1)
            tokenizer = fake_tokenizer()
            processor = Mock()
            processor.from_pretrained.return_value.tokenizer = tokenizer
            with patch.object(reward.user_profile_rules, "language_tools"), \
                    patch.object(reward, "judge", side_effect=judge), \
                    patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoProcessor=processor)}):
                scorer = reward.L1Reward(embedder=FakeEmbedder(), tokenizer=tokenizer)
                with patch.object(layer1_sampling, "L1Reward", return_value=scorer) as constructor, patch("builtins.print"):
                    layer1_sampling.score(args)
            processor.from_pretrained.assert_called_once_with("generation-checkpoint", trust_remote_code=True)
            constructor.assert_called_once_with("synthetic", 256, embedding_url=None, tokenizer=tokenizer)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["metrics"]["past_assignment"]["mean"], 0.5)
            self.assertNotIn("past_recall@1", summary["metrics"])
            self.assertEqual(summary["metrics"]["future_retrieval"]["mean"], 0.5)
            self.assertNotIn("future_recall@1", summary["metrics"])
            self.assertIn("past_owners", (root / "report.html").read_text())

    def test_from_val_attaches_action_frequencies(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from case_study import layer1_sampling

        extra = {"index": 0, "user_id": "u0", "content": content(["P0", "P1"]), "prompt_tokens": 2,
                 "signals_dropped": 0, "future": [{"date": "2026-10-10", "source": "Bing", "action": "F0"}],
                 "past_freq": [7, 9], "future_freq": [3]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist([{"extra_info": extra, "reward_model": {"ground_truth": ["F0"]}}]),
                           root / "val.parquet")
            (root / "0.jsonl").write_text(json.dumps({"input": "chat " + extra["content"], "output": answer(1),
                                                      "step": 0}) + "\n")
            args = argparse.Namespace(val_data=root / "val.parquet", dump=root / "0.jsonl", output_dir=root / "out",
                                      temperature=1.0, top_p=1.0)
            with patch("builtins.print"):
                layer1_sampling.from_val(args)
            row = json.loads((root / "out" / "rollouts.jsonl").read_text())
        self.assertEqual(layer1_sampling.action_freqs(row), ([7, 9], [3]))
        self.assertEqual([s["action"] for s in row["signals"]], ["P0", "P1"])
        with self.assertRaisesRegex(SystemExit, "predates the reward's action weights"):
            layer1_sampling.action_freqs({**row, "future": [{"action": "F0"}]})

    def test_trend_reads_old_and_new_dumps(self):
        from case_study import layer1_sampling

        base = {"input": "synthetic prompt", "output": "synthetic answer", "score": 0.5,
                "json_valid": 1, "rule_pass": 1, "n_interests": 1,
                "coverage": 0.5}
        top_k = {"future_recall": 0.5, "future_precision": 0.5}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, extra in (("old", top_k), ("batch_pool", {"past_recall": 0.4, "past_precision": 0.6, **top_k}),
                                ("assignment", {"past_assignment": 0.7, "future_retrieval": 0.6})):
                validation = root / name
                validation.mkdir()
                (validation / "0.jsonl").write_text(json.dumps({**base, **extra, "step": 0}) + "\n")
                output = root / f"{name}-output"
                with patch("builtins.print"):
                    layer1_sampling.trend(argparse.Namespace(validation_dir=validation, output_dir=output))
                step = json.loads((output / "trend.json").read_text())[0]
                self.assertEqual("past_recall" in step, name == "batch_pool")
                self.assertEqual("past_assignment" in step, name == "assignment")
                self.assertEqual(step.get("future_recall"), None if name == "assignment" else 0.5)
                self.assertEqual(step.get("future_retrieval"), 0.6 if name == "assignment" else None)


class Layer1InferenceTest(unittest.TestCase):
    def test_prepare_reads_test_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from case_study import layer1_inference

        def behaviors(actions, date):
            return [{"date": date, "source": "Bing", "action": action, "detailed_source": ""} for action in actions]

        rows = [{"user_id": f"user{index}", "past_behaviors": behaviors([f"P{index}"], "2026-07-10"),
                 "future_behaviors": behaviors([f"F{index}"] if index else [], "2026-07-17")} for index in range(3)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.Table.from_pylist(rows), root / "test-00000-of-00001.parquet")
            args = argparse.Namespace(data_dir=root, output_dir=root / "out", seed=0, num_prompts=0)
            with patch("builtins.print"):
                layer1_inference.prepare(args)
            prompts = [json.loads(line) for line in (root / "out" / "prompts.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(prompt["user_id"] for prompt in prompts), ["user1", "user2"])  # user0: no future
        self.assertEqual([prompt["prompt_id"] for prompt in prompts], [0, 1])
        for prompt in prompts:
            index = prompt["user_id"][-1]
            self.assertEqual([signal["action"] for signal in prompt["signals"]], [f"P{index}"])
            self.assertEqual([behavior["action"] for behavior in prompt["future"]], [f"F{index}"])

    def test_report_checks_rules_without_embeddings(self):
        from case_study import layer1_inference

        with TemporaryDirectory() as directory:
            root = Path(directory)
            signals = [{"date": "2026-10-03", "source": "Bing", "action": "P0"}]
            rows = [{"prompt_id": index, "user_id": f"user{index}", "content": content(["P0"]), "signals": signals,
                     "future": [{"date": "2026-10-10", "source": "Bing", "action": "F0"}],
                     "prompt_tokens": 2, "signals_dropped": 0,
                     "samples": [{"text": text, "finish_reason": "stop", "output_tokens": 7}]}
                    for index, text in enumerate((answer(2), "not JSON"))]
            (root / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (root / "rollout_settings.json").write_text(json.dumps({"checkpoint": "generation-checkpoint", "n": 1}))
            processor = Mock()
            processor.from_pretrained.return_value.tokenizer = fake_tokenizer()
            with patch.object(layer1_inference, "judge", side_effect=judge), \
                    patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoProcessor=processor)}), \
                    patch("builtins.print"):
                layer1_inference.report(argparse.Namespace(output_dir=root, checkpoint=None))
            processor.from_pretrained.assert_called_once_with("generation-checkpoint", trust_remote_code=True)
            summary = json.loads((root / "summary.json").read_text())
            page = (root / "report.html").read_text(encoding="utf-8")
            self.assertFalse((root / "scores.jsonl").exists())
        self.assertEqual((summary["prompts"], summary["json_valid"], summary["rule_pass"]), (2, 0.5, 0.5))
        self.assertEqual(summary["n_interests_valid"]["max"], 2)
        self.assertIn('"future_hits":null', page)
        self.assertIn('"output_tokens":7', page)

if __name__ == "__main__":
    unittest.main()
