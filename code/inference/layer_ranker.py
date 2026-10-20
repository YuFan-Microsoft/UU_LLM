"""Ranker after layer3_postprocessing: one model call per user (V2_User_Profile_Ranker, prompt_lr.md) picks the 10
candidate interests whose queries should retrieve the most future activities, best first. The answer's output_idx is
mapped back to the interests, kept in the layer3_postprocessing record as "ranked_interests", and that record is
written here (layer3.run leaves it to this step).

The input is built exactly as code/evaluation/ranker_data_generation.py builds the training data.
"""

import random

from task import Task, UserProfile, emit_record, output
from utils import RANKER_TOP_K, ranker_keys_valid

# ranker_data_generation.py defaults: --min_commercial_score medium and INPUT_FIELDS.
RANKER_COMMERCIAL_SCORES = {"medium", "high"}
RANKER_INPUT_FIELDS = ("interest_name", "actual_activity", "inferred_intent", "brands", "retailers", "products",
                       "predicted_queries", "first_detect_date", "last_detect_date", "topics")


def clean_queries(queries) -> list[str]:
    """evaluate_behavior_retrieval.clean_queries: the stripped non-empty queries, exact repeats removed."""
    if not isinstance(queries, list):
        return []
    seen, result = set(), []
    for query in queries:
        if isinstance(query, str) and (query := query.strip()) and query not in seen:
            seen.add(query)
            result.append(query)
    return result


class Ranker(Task):
    """V2_User_Profile_Ranker: the 10 candidate interests whose queries should retrieve the most future activities,
    best first. Its answer is added to the layer3_postprocessing record (no record of its own)."""
    stage, layer, prompt_file = "ranker", "layer3_postprocessing", "prompt_lr.md"

    def keys_valid(self, output: dict) -> bool:
        return ranker_keys_valid(output)

    @staticmethod
    def candidates(layer3: dict) -> list[dict]:
        """ranker_data_generation.candidate_interests before the shuffle: the layer3_postprocessing interests (sorted
        by confidence) that are commercial with a commercial_score of at least medium and have a non-blank query."""
        return [i for i in layer3.get("interests", []) if isinstance(i, dict) and i.get("commercial") is True
                and isinstance(i.get("commercial_score"), str)
                and i["commercial_score"].strip().casefold() in RANKER_COMMERCIAL_SCORES
                and clean_queries(i.get("predicted_queries"))]

    @staticmethod
    def shuffled(candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        """The candidates in input order: one shuffle seeded by (seed, user_id), as ranker_data_generation does."""
        order = list(candidates)
        random.Random(f"{seed}:{user_id}").shuffle(order)
        return order

    @staticmethod
    def project(interest: dict) -> dict:
        """ranker_data_generation.project_interest: INPUT_FIELDS only, cleaned queries, topic names without
        evidence."""
        projected = {key: interest.get(key) for key in RANKER_INPUT_FIELDS}
        projected["predicted_queries"] = clean_queries(interest.get("predicted_queries"))
        projected["topics"] = [{"topic": topic.get("topic")} for topic in interest.get("topics") or []]
        return projected

    def payload(self, candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        return [{"idx": index, **self.project(interest)}
                for index, interest in enumerate(self.shuffled(candidates, user_id, seed))]

    @staticmethod
    def check(count: int):
        """Exactly RANKER_TOP_K distinct idx of the `count` candidates (the trained keys and a usable ranking)."""
        def valid(answer: dict) -> bool:
            idx = answer.get("output_idx") if ranker_keys_valid(answer) else None
            return (idx is not None and len(idx) == RANKER_TOP_K and len(set(idx)) == RANKER_TOP_K
                    and all(0 <= i < count for i in idx))
        return valid

    def ranked(self, answer: dict | None, candidates: list[dict], user_id: str, seed: int) -> list[dict]:
        """The layer3_postprocessing interests named by output_idx, best first; [] without a valid answer."""
        if not answer:
            return []
        order = self.shuffled(candidates, user_id, seed)
        return [dict(order[index]) for index in answer["output_idx"]]


RANKER = Ranker()
TASKS = [RANKER]


def run(profiles: list[UserProfile], encode, run_round, emit, seed: int = 0) -> None:
    """One Ranker round on every user's layer3_postprocessing (profile.layer3, set by layer3.run); adds
    "ranked_interests" to it and writes the record with the ranker call.

    Only a user with more than RANKER_TOP_K candidates gets a call (with exactly RANKER_TOP_K the model could only
    reorder them). A user with at most RANKER_TOP_K candidates, or whose prompt only fits that many, gets no call: its
    "ranked_interests" are the first RANKER_TOP_K candidates in confidence order. That of a user whose answer never
    validated is []. Layer 4 reads only "interests", so the ranking does not change its inputs."""
    requests = {}
    for profile in profiles:
        candidates = RANKER.candidates(profile.layer3)
        item = None
        if len(candidates) > RANKER_TOP_K:
            # Too long a prompt leaves out the lowest-confidence candidates before the shuffle.
            item = RANKER.request(profile, lambda k: RANKER.payload(candidates[:k], profile.user_id, seed),
                                  len(candidates), encode, make_check=RANKER.check)
            if item and item["kept"] <= RANKER_TOP_K:
                item = None
        requests[profile.user_id] = (candidates, item)
    run_round([item for _, item in requests.values() if item])

    for profile in profiles:
        candidates, item = requests[profile.user_id]
        profile.layer3["ranked_interests"] = (
            RANKER.ranked(output(item), candidates[:item["kept"]], profile.user_id, seed) if item
            else [dict(interest) for interest in candidates[:RANKER_TOP_K]])
        emit_record(emit, profile, profile.layer3, item)
