"""Shared base for the Layer-3 / Layer-4 model tasks (one class per task in layer3.py / layer4.py) and the per-user
state they pass along."""

from dataclasses import dataclass, field

from utils import HERE, INPUT_MARKER, PROMPT_BUDGET, dumps, new_item, query_language

PROMPT_DIR = HERE / "prompts"


def exact_list_keys(value, keys) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) and set(item) == keys for item in value)


def exact_dict_keys(value, keys) -> bool:
    return isinstance(value, dict) and set(value) == keys


def output_gate(interest: dict) -> bool:
    """layer2_gates.output_gate: fine interests need confidence >= 0.2."""
    conf = float(interest.get("confidence_score") or 0)
    if interest.get("interest_type") == "coarse":
        return len(interest.get("children", [])) > 0 and conf > 0.01
    return conf >= 0.2


def normalize_category(category) -> str:
    """A canonical "/A/B" category path (layer3_persona / layer4_hyper_commercial_interest _normalize_category)."""
    if not isinstance(category, str):
        return ""
    parts = [part.strip() for part in category.strip().split("/") if part.strip()]
    return f"/{'/'.join(parts)}" if parts else ""


@dataclass
class UserProfile:
    """One user's final Layer-2 snapshot and the Layer-3 / Layer-4 records built from it. Like the production run
    (Layer 3 once on the last window with --force-refresh), every user runs on `run_date`, the last grid window:
    the task records carry that date, while the postprocessing records keep the snapshot's own fields (its `date`,
    and `_carried_forward` for users idle in the last window), as maiprofilev3dev copies the snapshot."""
    user_id: str
    snapshot: dict
    run_date: str
    layer3: dict = field(default_factory=dict)       # layer3_postprocessing
    biography: dict = field(default_factory=dict)    # layer4_biography
    preference: dict = field(default_factory=dict)   # layer4_commercial_preference

    @property
    def date(self) -> str:
        return self.run_date

    @property
    def language(self) -> dict:
        """resolve_query_language(...).state() without a user Market: {"locale", "source"}."""
        return query_language(self.snapshot.get("predicted_content_locale"))


class Task:
    """One model task: its prompt, the answer key check used for retries, and the record it produces."""
    stage = ""        # name in predictions.jsonl; also seeds sampling
    layer = ""        # maiprofilev3dev step_key of the record
    prompt_file = ""

    def __init__(self) -> None:
        self.prompt = (PROMPT_DIR / self.prompt_file).read_text(encoding="utf-8")

    def keys_valid(self, output: dict) -> bool:
        """user_profile_rules keys_valid of this task: the exact key sets of a trained answer."""
        raise NotImplementedError

    def message(self, payload: dict) -> str:
        return self.prompt + INPUT_MARKER + dumps(payload)

    def request(self, profile: UserProfile, build, count: int, encode, make_check=None, suffix: str = ""):
        """A model request whose payload is build(k) for the largest k <= count (items sorted by confidence, so the
        lowest-confidence ones are left out) that fits the prompt budget; None when not even one item fits.
        make_check(k) gives a validator for that payload (default: keys_valid)."""
        k = count
        while True:
            prompt_ids = encode(self.message(build(k)))
            if len(prompt_ids) <= PROMPT_BUDGET or k == 0:
                break
            k = max(0, min(k - 1, int(k * PROMPT_BUDGET / len(prompt_ids) * 0.95)))
        if not k:
            return None
        return {**new_item(f"{profile.user_id}|{profile.date}{suffix}", prompt_ids), "stage": self.stage,
                "check": make_check(k) if make_check else self.keys_valid, "user_id": profile.user_id,
                "kept": k, "dropped": count - k}

    def record(self, profile: UserProfile, **fields) -> dict:
        return {"user_id": profile.user_id, "date": profile.date, "layer": self.layer, **fields}


def output(item: dict | None) -> dict | None:
    """The valid answer of a request, or None (no request, or never valid)."""
    return item["output"] if item else None


def emit_record(emit, profile: UserProfile, record: dict, *items) -> None:
    """Write a record and the calls (model requests) behind it."""
    emit(kind="record", layer=record["layer"], date=profile.date, record=record)
    for item in items:
        if item:
            emit(kind="call", stage=item["stage"], user_id=profile.user_id, date=profile.date,
                 attempts=item["attempts"], valid=item["output"] is not None, dropped=item["dropped"],
                 text=item["text"])
