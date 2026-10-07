#!/usr/bin/env python3
"""Build Layer 3 / Layer 4 SFT train/test JSONL in the user_profile_dataset schema.

One config per task, each row:
    {"uid": sha256(user_id)[:16],
     "messages": [{"role": "user", "content": PROMPT + "\\nInput:\\n" + <input JSON>},
                  {"role": "assistant", "content": <output JSON>}],
     "source": "GPT54", "domain": "maiprofile-<task>", "think_type": "none", "language": <see LANGUAGE>}

PROMPT comes from UU_LLM/prompts/<prompt file>. The input JSON carries only per-user data (no user_id or date);
JSON is minified. Train/test is split by user with the layer1_step3 rule (bucket = uid % 10000 < threshold).
Upstream tasks use a threshold at least as large as their downstream tasks, so a downstream test user's upstream
rows are also test rows, and every config keeps at least 2,000 test rows.

Writes <out-dir>/<config>/{train-0000i-of-0000n,test-00000-of-00001}.jsonl and prints row/user counts.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import orjson
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
PROCESS = ROOT / "ProcessData"
PROMPTS = ROOT / "UU_LLM" / "prompts"
INPUT_MARKER = "\nInput:\n"
SHARD_BYTES = 1_000_000_000


def l3_persona(i: dict, o: dict) -> tuple[dict, dict]:
    return {"facts": i.get("facts") or {}, "interests": i["active_interests"]}, o


def l3_commercial(i: dict, o: dict) -> tuple[dict, dict]:
    return {"interests": i["interests"], "query_language": i["query_language"]}, o


def l4_biography(i: dict, o: dict) -> tuple[dict, dict]:
    return {"facts": i.get("facts") or {}, "interests": i["active_interests"]}, o


def l4_commercial_preference(i: dict, o: dict) -> tuple[dict, dict]:
    keys = ("life_stage", "commercial_interests", "non_commercial_interest_names")
    return {key: i[key] for key in keys}, o


def l4_mission_discovery(i: dict, o: dict) -> tuple[dict, dict]:
    keys = ("personal_context", "professional_context", "commercial_preferences", "commercial_interests")
    return {key: i[key] for key in keys}, o


def l4_mission_enhancement(i: dict, o: dict) -> tuple[dict, dict]:
    keys = ("query_language", "candidate_missions", "source_evidence", "personal_context", "professional_context",
            "commercial_preferences", "world_knowledge")
    return {key: i[key] for key in keys}, o


# config -> (cleaned input, prompt file, domain, input builder, test threshold per 10,000, language field or None)
TASKS = {
    "User_Profile_L3_Persona_gpt54_V1": (
        "l3_persona/step2_rule_based_clean.jsonl", "prompt_l3_persona.md", "maiprofile-layer3-persona",
        l3_persona, 1230, None),
    "User_Profile_L3_Commercial_gpt54_V1": (
        "l3_commercial/step3_rule_based_clean.jsonl", "prompt_l3_commercial.md", "maiprofile-layer3-commercial",
        l3_commercial, 1230, "query_language"),
    "User_Profile_L4_Biography_gpt54_V1": (
        "l4_biography/step2_rule_based_clean.jsonl", "prompt_l4_biography.md", "maiprofile-layer4-biography",
        l4_biography, 1230, None),
    "User_Profile_L4_CommercialPreference_gpt54_V1": (
        "l4_commercial_preference/step2_rule_based_clean.jsonl", "prompt_l4_commercial_preference.md",
        "maiprofile-layer4-commercial-preference", l4_commercial_preference, 1230, None),
    "User_Profile_L4_MissionDiscovery_gpt54_V1": (
        "l4_hyper_mission_discovery/step2_rule_based_clean.jsonl", "prompt_l4_hyper_mission_discovery.md",
        "maiprofile-layer4-mission-discovery", l4_mission_discovery, 1010, None),
    "User_Profile_L4_MissionEnhancement_gpt54_V1": (
        "l4_hyper_mission_enhancement/step3_rule_based_clean.jsonl", "prompt_l4_hyper_mission_enhancement.md",
        "maiprofile-layer4-mission-enhancement", l4_mission_enhancement, 200, "query_language"),
}


def dumps(value) -> str:
    return orjson.dumps(value).decode("utf-8")


def user_hash(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def build(config: str, out_dir: Path) -> dict:
    source, prompt_file, domain, build_input, threshold, language_key = TASKS[config]
    prompt = (PROMPTS / prompt_file).read_text(encoding="utf-8")
    rows = {"train": [], "test": []}
    users = {"train": set(), "test": set()}
    with (PROCESS / source).open("rb") as handle:
        for line in tqdm(handle, desc=config, unit="rows", leave=False):
            record = orjson.loads(line)
            payload, answer = build_input(record["input"], record["output"])
            uid = user_hash(record["input"]["user_id"])
            split = "test" if int(uid, 16) % 10_000 < threshold else "train"
            example = {
                "uid": uid,
                "messages": [{"role": "user", "content": prompt + INPUT_MARKER + dumps(payload)},
                             {"role": "assistant", "content": dumps(answer)}],
                "source": "GPT54",
                "domain": domain,
                "think_type": "none",
                "language": record["input"][language_key] if language_key else "en",
            }
            rows[split].append(orjson.dumps(example) + b"\n")
            users[split].add(uid)

    target = out_dir / config
    target.mkdir(parents=True, exist_ok=True)
    for split, lines in rows.items():
        shards = 1 if split == "test" else max(1, math.ceil(sum(map(len, lines)) / SHARD_BYTES))
        size = math.ceil(len(lines) / shards)
        for index in range(shards):
            path = target / f"{split}-{index:05d}-of-{shards:05d}.jsonl"
            path.write_bytes(b"".join(lines[index * size:(index + 1) * size]))
    return {split: {"rows": len(lines), "users": len(users[split])} for split, lines in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Layer 3 / Layer 4 SFT configs.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory that receives one folder per config")
    parser.add_argument("--configs", nargs="+", default=list(TASKS), choices=list(TASKS), help="Configs to build")
    args = parser.parse_args()
    for config in args.configs:
        counts = build(config, args.out_dir)
        print(f"{config}: train {counts['train']['rows']:,} rows / {counts['train']['users']:,} users, "
              f"test {counts['test']['rows']:,} rows / {counts['test']['users']:,} users", flush=True)


if __name__ == "__main__":
    main()
