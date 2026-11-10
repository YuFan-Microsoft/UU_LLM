"""The Layer-1 RL data shared by prepare_data.py (GRPO training data) and the case study (case_study/).

RL_User_Profile_Layer1_V1 (folder User_Profile_Layer1_For_RL_V1) of the gated yufan/user_profile_dataset: Parquet
shards, one row per user, {"user_id", "past_behaviors": [{"date", "source", "action", "action_idx",
"action_freq"}], "future_behaviors": [...], "language"}, in two splits with disjoint users: train (197,269 rows in 40
shards) and test (1,976 rows). The history is one window of the user and the future the actions after it (at least
10). GRPO trains on train and validates on test users; the case study reads test users. Both windows are cleaned as
code/inference does for one window, and every tool takes rows in the seeded random order of row_key over the user_id
(one row per user, so the order does not depend on the file layout): for one seed the case study's prompts are (up to
the rows each tool skips) the GRPO run's validation users.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import tempfile

UU_LLM = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(UU_LLM / "code" / "inference"))

from utils import MAX_ACTION_CHARS, dedupe, load_users  # noqa: E402

# The Layer-1 GRPO starting point: the VFinal F16A unified SFT model, Qwen3.5-4B.
DEFAULT_CHECKPOINT = "/yufan/MAI_Profile/checkpoints/unified_slm_VFinal_F16A/4B/Final_CKPT"
HF_REPO = "yufan/user_profile_dataset"
HF_CONFIG_NAME = "RL_User_Profile_Layer1_V1"
HF_CONFIG = "User_Profile_Layer1_For_RL_V1"  # the folder of HF_CONFIG_NAME in the repository
COLUMNS = ["user_id", "past_behaviors", "future_behaviors", "language"]
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
    """Download the Parquet shards of one RL_User_Profile_Layer1_V1 split ("train" or "test") into `directory`."""
    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    files = sorted(name for name in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if name.startswith(f"{HF_CONFIG}/{split}-") and name.endswith(".parquet"))
    if not files:
        raise SystemExit(f"No {HF_CONFIG}/{split}-*.parquet in {HF_REPO}")
    print(f"Downloading {len(files)} {split} file(s) of {HF_REPO}/{HF_CONFIG} to a temporary directory", flush=True)
    return [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", token=token, local_dir=directory))
            for name in files]


@contextmanager
def rl_data_files(data_dir: Path | None, split: str):
    """The Parquet files of a split: the local copies in `data_dir` (<split>-*.parquet, as in the repository folder),
    else downloaded to a temporary directory (deleted on exit)."""
    if data_dir:
        paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not paths:
            raise SystemExit(f"No {split}-*.parquet under {data_dir}")
        yield paths
    else:
        with tempfile.TemporaryDirectory(prefix=f"l1_rl_{split}_") as directory:
            yield download_rl_data(Path(directory), split)


def row_key(seed: int, data: bytes) -> int:
    """The seeded hash of `data` (a user_id, encoded): rows by increasing key are in a random order fixed for the
    seed, independent of the file order and location."""
    return int.from_bytes(hashlib.blake2b(f"{seed}|".encode() + data, digest_size=8).digest(), "big")


def index_rows(paths: list[Path], seed: int) -> list[tuple[str, int]]:
    """(path, row number in the file) of every row, in the seeded order of row_key over its user_id; reads the
    user_id column only. One row per user is required (the key is the user)."""
    import pyarrow.parquet as pq

    keyed, users = [], set()
    for path in paths:
        for number, user_id in enumerate(pq.read_table(path, columns=["user_id"]).column(0).to_pylist()):
            if user_id in users:
                raise SystemExit(f"{path}: user {user_id} has a second row; expected one row per user_id")
            users.add(user_id)
            keyed.append((row_key(seed, user_id.encode()), str(path), number))
    keyed.sort()
    return [(path, number) for _, path, number in keyed]


def read_rows(path: str | Path, numbers: list[int]) -> list[dict]:
    """The rows (COLUMNS) at these row numbers of one Parquet file, in the given order."""
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=COLUMNS).take(numbers).to_pylist()


def clean_behaviors(user_id: str, behaviors: list[dict]) -> list[dict]:
    """The inference cleaning of one weekly window: load_users (128-char actions, dates >= 2025-01-01, sorted by
    date) then dedupe (allowed sources only, one signal per action). Each kept behavior gets back the action_freq of
    its raw behaviors (the reward's action weight; None when the data has none), which the Layer-1 message omits."""
    freqs: dict[str, int] = {}
    for behavior in behaviors:
        action = (behavior.get("action") or "")[:MAX_ACTION_CHARS]
        if behavior.get("action_freq") is not None:
            freqs[action] = max(freqs.get(action, 0), int(behavior["action_freq"]))
    users = load_users([{"user_id": user_id, "past_behaviors": behaviors}])
    return [{**behavior, "action_freq": freqs.get(behavior["action"])}
            for behavior in dedupe(users[user_id])] if users else []
