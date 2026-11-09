"""Layer-3 Commercial GRPO reward: the JEV classifier's judgement of the rollout's commercial profile.

The reward of one answer: 0 if it breaks the Layer-3 Commercial rules (user_profile_rules.score_example, the SFT
checks: not V3 JSON, interest names not exactly the input's, invalid score / stage / query values, duplicates, query
format or language; a cut answer is not valid JSON), else the mean over the user's target future behaviors (at most
20, fixed per user by prepare_rl_data.py, so every answer of a GRPO group is scored on the same ones) of the JEV
expected match
    (0 * P(no match) + 1 * P(partial match) + 2 * P(perfect match)) / 2      in [0, 1]
for the answer's mai_profile (jev_profile.build_mai_profile: the user's reference commercial interests, the first 10
in input order, with the answer's brands / products / retailers / predicted queries, their category and topics). The
reference interests are those a fixed checkpoint marks commercial with a medium or high score (prepare_rl_data.py,
code/inference/tag_commercial_interests.py), not the rollout's own flags: marking more interests commercial cannot
raise the reward (without a reference, the answer's flags are used).

SIGNALS, logged next to the reward (the reward does not use them): rule_pass, and two reward-hacking watches,
commercial_share (the input interests the answer marks commercial: free to drift, the reward ignores it) and
entities_per_interest (brands + products + retailers per profile interest); both None for an answer that is not
valid JSON.

    reward_fn = L3Reward("http://node-2:8201,http://node-2:8202")   # serve_jev.sh's last line
    reward_fn(prompt, category, answer, targets, reference)["reward"]   # reference: the commercial interest names
"""

import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_profile import TOP_K, JEVClient, build_mai_profile, prompt_interests  # noqa: E402  (puts inference first)
import user_profile_rules  # noqa: E402
from utils import parse_json_object  # noqa: E402

LABEL_WEIGHTS = {"no match": 0.0, "partial match": 0.5, "perfect match": 1.0}  # 0 / 1 / 2, over 2
SIGNALS = ("rule_pass", "commercial_share", "entities_per_interest")
ENTITY_KEYS = ("brands", "products", "retailers")


def hack_signals(prompt: str, output: dict, reference: list[str] | None = None) -> dict:
    """commercial_share and entities_per_interest of a JSON-valid answer (None when undefined); the profile
    interests are the reference ones when given."""
    interests = prompt_interests(prompt)
    entries = {}
    for entry in output["interest_commercial"]:
        entries.setdefault(entry["interest_name"], entry)
    commercial = [entries[i["interest_name"]] for i in interests
                  if i["interest_name"] in entries and entries[i["interest_name"]]["commercial"] is True]
    profile = commercial[:TOP_K] if reference is None else [
        entries[i["interest_name"]] for i in interests
        if i["interest_name"] in entries and i["interest_name"] in reference][:TOP_K]
    return {"commercial_share": len(commercial) / len(interests) if interests else None,
            "entities_per_interest": statistics.mean(sum(len(entry[key]) for key in ENTITY_KEYS) for entry in profile)
            if profile else None}


class L3Reward:
    def __init__(self, jev_urls):
        user_profile_rules.language_tools()  # the language rule needs fastText lid.176.bin: fail before scoring
        self.client = JEVClient(jev_urls)
        self.lock = threading.Lock()  # the rule checks (fastText) one at a time; the JEV requests run concurrently

    def __call__(self, prompt: str, category: str, answer: str, targets: list[str],
                 reference: list[str] | None = None) -> dict:
        result = {"reward": 0.0, "rule_pass": False, "commercial_share": None, "entities_per_interest": None}
        with self.lock:
            rules = user_profile_rules.score_example([{"role": "user", "content": prompt}], answer)
            if rules["json_valid"]:
                result.update(hack_signals(prompt, parse_json_object(answer), reference))
        result["rule_pass"] = rules["rule_pass"]
        if not rules["rule_pass"] or not targets:
            return result
        profile = build_mai_profile(prompt, category, answer, reference=reference)
        probs = self.client.score([(profile, target) for target in targets])
        result["reward"] = statistics.mean(sum(weight * p[label] for label, weight in LABEL_WEIGHTS.items())
                                           for p in probs)
        return result
