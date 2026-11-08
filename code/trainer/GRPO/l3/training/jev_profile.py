"""Layer-3 Commercial rollout -> EVAL_Dataset mai_profile -> JEV probabilities of the user's future behaviors.

One RL_User_Profile_Layer3_V1_Processed row (prompt, category, future_behaviors) and one rollout answer of its prompt:
  1. the prompt's input interests in their order (the Layer-2 confidence order), each matched by exact interest_name
     to the answer's first entry of that name; of those the answer marks "commercial": true, the first TOP_K (10)
  2. each becomes {"Interest": name, "Category": category[name], "Brand" / "Product" / "Retailer": the answer's
     brands / products / retailers, "Query": its predicted_queries, "Topic": the prompt's topics of that interest}
  3. mai_profile() of ../../../../evaluation/e2e_ads_eval/classify.py turns them into the EVAL_Dataset string the
     JEV classifier was trained on (same items, same order)
The input order, not the answer's, picks the top TOP_K, so reordering the answer changes nothing; answer entries
whose name is not an input interest are ignored. An answer that is not valid Layer-3 Commercial JSON has no profile
(None); one without a commercial interest has the empty profile "[]", which is scored like any other.

    from jev_profile import JEVClient, build_mai_profile, score_profiles
    profiles = [build_mai_profile(row["prompt"], row["category"], answer) for row, answer in rollouts]
    probs = score_profiles(JEVClient(jev_urls), profiles, [row["future_behaviors"] for row, _ in rollouts])
    # jev_urls: serve_jev.sh's last line, one URL per GPU; the batch's pairs are spread over all of them
    # probs[i]: None, or one {"no match": p, "partial match": p, "perfect match": p} per future behavior of rollout i
"""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[4]
for directory in (CODE / "inference", CODE / "evaluation" / "e2e_ads_eval", CODE / "trainer" / "GRPO" / "jev"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from classify import mai_profile  # noqa: E402
from jev_client import JEVClient  # noqa: E402,F401  (re-exported for the reward)
from utils import l3_commercial_keys_valid, parse_json_object  # noqa: E402

TOP_K = 10
INPUT_MARKER = "\nInput:\n"  # prompt = prompt_l3_commercial.md + INPUT_MARKER + the minimized interest payload


def prompt_interests(prompt: str) -> list[dict]:
    """The Commercial input interests of a prompt, in their order."""
    return json.loads(prompt[prompt.rindex(INPUT_MARKER) + len(INPUT_MARKER):])["interests"]


def build_mai_profile(prompt: str, category: str, answer: str, top_k: int = TOP_K) -> str | None:
    """The mai_profile string of one rollout answer (see the module docstring); None when the answer is invalid."""
    output = parse_json_object(answer)
    if output is None or not l3_commercial_keys_valid(output):
        return None
    entries = {}
    for entry in output["interest_commercial"]:
        entries.setdefault(entry["interest_name"], entry)
    categories = json.loads(category)
    interests = []
    for interest in prompt_interests(prompt):
        name = interest["interest_name"]
        entry = entries.get(name)
        if entry is None or entry["commercial"] is not True:
            continue
        interests.append({"Interest": name, "Category": categories.get(name, ""),
                          "Brand": entry["brands"], "Product": entry["products"], "Retailer": entry["retailers"],
                          "Query": entry["predicted_queries"], "Topic": [t["topic"] for t in interest["topics"]]})
        if len(interests) == top_k:
            break
    return mai_profile(interests)


def score_profiles(client: JEVClient, profiles: list[str | None],
                   future_behaviors: list[list[str]]) -> list[list[dict] | None]:
    """Per rollout: None when its profile is None, else the JEV probabilities of each of its future behaviors, in
    order. Every distinct (profile, behavior) pair of the batch is scored once, all in one request."""
    pairs = list(dict.fromkeys((profile, behavior) for profile, behaviors in zip(profiles, future_behaviors)
                               if profile is not None for behavior in behaviors))
    scores = dict(zip(pairs, client.score(pairs))) if pairs else {}
    return [None if profile is None else [scores[(profile, behavior)] for behavior in behaviors]
            for profile, behaviors in zip(profiles, future_behaviors)]
