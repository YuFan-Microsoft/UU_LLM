"""The Layer-1 RL data shared by prepare_data.py (GRPO training data) and case_study/case_study.py.

User_Profile_Weekly_For_RL of the gated yufan/user_profile_dataset: one JSONL row per user-week,
{"user_id", "past_behaviors": [{"source", "action", "date"}], "future_behaviors": [...]}, in two splits with
disjoint users: train (139,629 rows, 31,543 users) and test (2,000 rows of 456 held-out users). GRPO trains on train
and validates on test users; the case study reads test users. Both weeks are cleaned as code/inference does for one
window, and both tools take rows in the seeded random order of row_key, so for one seed the case study's prompts are
(up to the rows each tool skips) the GRPO run's validation users.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import tempfile

UU_LLM = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(UU_LLM / "code" / "inference"))

from utils import dedupe, load_users  # noqa: E402

# The Layer-1 GRPO starting point: the VFinal F16A unified SFT model, Qwen3.5-4B.
DEFAULT_CHECKPOINT = "/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT"
HF_REPO = "yufan/user_profile_dataset"
HF_CONFIG = "User_Profile_Weekly_For_RL"
# Key files, git-ignored: in UU_LLM, or in the directory above it (the job directory UU_LLM was copied into).
HF_TOKEN_FILES = tuple(directory / name for directory in (UU_LLM, UU_LLM.parent) for name in ("hf_token.txt", "HF_Key"))


def hf_token() -> str | None:
    """HF_TOKEN, else the first line of the first existing HF_TOKEN_FILES."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for path in HF_TOKEN_FILES:
        if path.exists() and path.read_text(encoding="utf-8").split():
            return path.read_text(encoding="utf-8").split()[0]
    return None


def download_rl_data(directory: Path, split: str) -> list[Path]:
    """Download the JSONL shards of one User_Profile_Weekly_For_RL split ("train" or "test") into `directory`."""
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    files = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/{split}-") and name.endswith(".jsonl"))
    if not files:
        raise SystemExit(f"No {HF_CONFIG}/{split}-*.jsonl in {HF_REPO}")
    print(f"Downloading {len(files)} {split} file(s) of {HF_REPO}/{HF_CONFIG} to a temporary directory", flush=True)
    return [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
            for name in files]


@contextmanager
def rl_data_files(data_dir: Path | None, split: str):
    """The JSONL files of a split: the local copies in `data_dir` (<split>-*.jsonl, as on the Hub), else downloaded to
    a temporary directory (deleted on exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{split}-*.jsonl"))
        if not paths:
            raise SystemExit(f"No {split}-*.jsonl under {data_dir}")
        yield paths
    else:
        with tempfile.TemporaryDirectory(prefix=f"l1_rl_{split}_") as directory:
            yield download_rl_data(Path(directory), split)


def iter_lines(paths: list[Path]):
    """(file name, line number, raw line) of every non-empty row."""
    for path in paths:
        with path.open("rb") as source:
            for line_number, line in enumerate(source):
                if line.strip():
                    yield path.name, line_number, line


def row_key(seed: int, line: bytes) -> int:
    """The seeded hash of a raw row: rows by increasing key are in a random order fixed for the seed, independent of
    the file order and location."""
    return int.from_bytes(hashlib.blake2b(f"{seed}|".encode() + line, digest_size=8).digest(), "big")


def clean_behaviors(user_id: str, behaviors: list[dict]) -> list[dict]:
    """The inference cleaning of one weekly window: load_users (128-char actions, dates >= 2025-01-01, sorted by
    date) then dedupe (allowed sources only, one signal per action)."""
    users = load_users([{"user_id": user_id, "past_behaviors": behaviors}])
    return dedupe(users[user_id]) if users else []
