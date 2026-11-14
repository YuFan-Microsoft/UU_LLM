"""Shared pieces of the Layer-3 Commercial best-of-N pipeline (see README.md).

Paths (code/inference, the GRPO L1 / L3 reward code, the JEV client) are put on sys.path here, so every step imports
the GRPO reward's own functions: the candidates are scored exactly as the GRPO L3 reward and the e2e ads eval score a
profile.
"""
import json
import os
import sys
from pathlib import Path

BON = Path(__file__).resolve().parent
CODE = BON.parents[1]
UU_LLM = CODE.parent
for directory in (CODE / "inference", CODE / "trainer" / "GRPO" / "l1", CODE / "trainer" / "GRPO" / "l3" / "training",
                  CODE / "trainer" / "GRPO" / "jev", CODE / "evaluation" / "e2e_ads_eval"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
# user_profile_rules finds fastText lid.176.bin here (downloaded when missing), as code/inference does.
os.environ.setdefault("LID_MODEL_PATH", str(UU_LLM / "models" / "lid.176.bin"))

HF_REPO = "yufan/user_profile_dataset"
# --dataset -> (HF config name, its folder in the repository). Same columns; "all" has every future behavior of the
# window, "commercial" only the commercial ones (product pages, ads, shopping).
DATASETS = {
    "all": ("RL_User_Profile_Layer3_Final_All", "User_Profile_Layer3_For_RL_Final_All"),
    "commercial": ("RL_User_Profile_Layer3_Final_Commercial", "User_Profile_Layer3_For_RL_V3_Final"),
}
DEFAULT_DATASET = "all"
COMMERCIAL_SCORE_RANK = {"low": 1, "medium": 2, "high": 3}
DEFAULT_MIN_COMMERCIAL_SCORE = "medium"  # as the e2e ads eval and evaluate_behavior_retrieval.py
MAX_MODEL_LEN = 25600  # prompt + answer, as the GRPO L3 run
MAX_NEW_TOKENS = 8192  # as the GRPO L3 run's max_response_length
MAX_PROMPT_LENGTH = MAX_MODEL_LEN - MAX_NEW_TOKENS


def hf_token() -> str | None:
    from rl_data import hf_token as token  # HF_TOKEN, else UU_LLM/HF_Key or hf_token.txt

    return token()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_uids(path: Path) -> set[str]:
    """The uids already written to a resumable JSONL output (a cut last line is ignored and rewritten)."""
    if not Path(path).exists():
        return set()
    uids = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            try:
                uids.add(json.loads(line)["uid"])
            except (ValueError, KeyError):
                pass
    return uids


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)
