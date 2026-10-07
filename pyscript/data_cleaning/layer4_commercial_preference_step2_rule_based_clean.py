#!/usr/bin/env python3
"""Rule-based cleaning of the Layer 4 commercial preference dataset (step1 output, one JSONL row per request).

Input/output row format (unchanged by this step):
    {"input":  {"user_id", "date", "life_stage": {...},
                "commercial_interests": [{"interest_name", "persona", "brands", "retailers", "products", "topics"}],
                "non_commercial_interest_names": [str]},
     "output": {"deal_seeking", "price_tier", "affinity"}}

A row is dropped when it falls into any CATEGORY:
  json_format           `output` does not match OUTPUT_SCHEMA (prompts/layer4_commercial_preference.md "Rules"
                        and "Output Format") or an object's keys are out of order:
       {"deal_seeking": {"value": "high"|"medium"|"low"|"unknown",
                         "details": [{"area", "seeking": "high"|"medium"|"low", "evidence", "signal_strength"}]},
        "price_tier":   {"value": "premium"|"mid"|"budget"|"unknown",
                         "details": [{"area", "tier": "premium"|"mid"|"budget", "evidence", "signal_strength"}]},
        "affinity": {
          "shopping": {"product_categories": [{"category": <Ads Vertical L1>, "description"}],
                       "shopper_type": {"value": [<archetype>],
                                        "details": [{"area", "type": <archetype>, "evidence", "signal_strength"}]}},
          "dining": {"restrictions": {"value": [<restriction label or "Unknown">],      # non-empty
                                      "details": [{"area", "restriction", "evidence", "signal_strength"}]}}}}
                        Every string is non-empty and signal_strength is "strong"|"moderate"|"weak".
  invalid_input         no commercial interests, an empty interest name, or a non-object life_stage
  inconsistent_content  a value contradicts its details as the prompt requires (deal_seeking, shopper_type,
                        dining restrictions), or product_categories repeats a category
  text_quality          the output repeats the user_id
FLAG_CATEGORIES are softer checks (prompt "Tips" that allow judgment) that are only counted on the kept rows.

Free text is not checked for braces: evidence legitimately quotes ad templates such as "{KeyWord:Craft Supplies}".
The JSON-format checker, cleaning loop, and summary printer come from layer4_biography_step2_rule_based_clean.py.

Writes the kept rows to <output> and prints the per-category summary table.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer4_biography_step2_rule_based_clean import (  # noqa: E402
    NON_EMPTY_STRING, CheckResult, clean, has_duplicates, has_empty_name, leaks_user_id, matches_format,
    parse_args, print_summary,
)


DEAL_LEVELS = ["high", "medium", "low"]
PRICE_TIERS = ["premium", "mid", "budget"]
SIGNAL_STRENGTHS = ["strong", "moderate", "weak"]
AD_VERTICALS_L1 = ["Apparel", "BeautyPersonalCare", "ComputersConsumerElectronics", "Finance", "Health",
                   "HomeGarden", "InternetTelecom", "RetailersGeneralMerchandise", "SportsFitness",
                   "TravelTourism", "Vehicles"]
SHOPPER_TYPES = ["ResearchDriven", "ImpulseDriven", "DealDriven", "PremiumDriven", "BrandRetailerLoyal",
                 "NoveltySeeker", "ConvenienceDriven", "ValueDriven"]
DINING_RESTRICTIONS = ["Vegan", "Vegetarian", "Keto", "Halal", "Kosher", "No Shellfish", "No Nuts", "No Gluten",
                       "Dairy-Free", "Low-Carb", "Paleo", "Pescatarian"]
UNKNOWN_RESTRICTION = "Unknown"


def detail_schema(level_key: str, levels: list[str]) -> dict:
    return {
        "type": "object",
        "required": ["area", level_key, "evidence", "signal_strength"],
        "additionalProperties": False,
        "properties": {
            "area": NON_EMPTY_STRING,
            level_key: {"enum": levels},
            "evidence": NON_EMPTY_STRING,
            "signal_strength": {"enum": SIGNAL_STRENGTHS},
        },
    }


def value_details_schema(value_schema: dict, details_item: dict) -> dict:
    return {
        "type": "object",
        "required": ["value", "details"],
        "additionalProperties": False,
        "properties": {"value": value_schema, "details": {"type": "array", "items": details_item}},
    }


def object_schema(properties: dict) -> dict:
    return {"type": "object", "required": list(properties), "additionalProperties": False, "properties": properties}


OUTPUT_SCHEMA = object_schema({
    "deal_seeking": value_details_schema({"enum": [*DEAL_LEVELS, "unknown"]}, detail_schema("seeking", DEAL_LEVELS)),
    "price_tier": value_details_schema({"enum": [*PRICE_TIERS, "unknown"]}, detail_schema("tier", PRICE_TIERS)),
    "affinity": object_schema({
        "shopping": object_schema({
            "product_categories": {"type": "array", "items": object_schema({
                "category": {"enum": AD_VERTICALS_L1},
                "description": NON_EMPTY_STRING,
            })},
            "shopper_type": value_details_schema(
                {"type": "array", "items": {"enum": SHOPPER_TYPES}},
                detail_schema("type", SHOPPER_TYPES)),
        }),
        "dining": object_schema({
            "restrictions": value_details_schema(
                {"type": "array", "minItems": 1, "items": {"enum": [*DINING_RESTRICTIONS, UNKNOWN_RESTRICTION]}},
                detail_schema("restriction", DINING_RESTRICTIONS)),
        }),
    }),
})
OUTPUT_VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)


def detail_order(level_key: str) -> tuple[str, ...]:
    return ("area", level_key, "evidence", "signal_strength")


KEY_ORDERS = {
    "": ("deal_seeking", "price_tier", "affinity"),
    "deal_seeking": ("value", "details"),
    "deal_seeking.details[]": detail_order("seeking"),
    "price_tier": ("value", "details"),
    "price_tier.details[]": detail_order("tier"),
    "affinity": ("shopping", "dining"),
    "affinity.shopping": ("product_categories", "shopper_type"),
    "affinity.shopping.product_categories[]": ("category", "description"),
    "affinity.shopping.shopper_type": ("value", "details"),
    "affinity.shopping.shopper_type.details[]": detail_order("type"),
    "affinity.dining": ("restrictions",),
    "affinity.dining.restrictions": ("value", "details"),
    "affinity.dining.restrictions.details[]": detail_order("restriction"),
}

CATEGORIES = {
    "json_format": "Output does not match the prompt's JSON format (missing/extra key, wrong type or enum value, "
                   "empty string, or keys out of order)",
    "invalid_input": "Input has no commercial interests, an empty interest name, or a non-object life_stage",
    "inconsistent_content": "A value contradicts its details (deal_seeking, shopper_type, dining restrictions) "
                            "or product_categories repeats a category",
    "text_quality": "The output repeats the user_id",
}
FLAG_CATEGORIES = {
    "weak_evidence": "price_tier is not the most frequent detail tier, premium/budget rests on one detail, "
                     "or deal_seeking is high without a strong detail",
    "empty_product_categories": "product_categories is empty",
    "duplicate_input_name": "Two input interests share the same interest_name",
}


def deal_seeking_consistent(deal: dict) -> bool:
    """unknown <=> no details; one detail level => that value; high and low details => medium."""
    value = deal["value"]
    levels = {detail["seeking"] for detail in deal["details"]}
    if value == "unknown" or not levels:
        return value == "unknown" and not levels
    if len(levels) == 1:
        return value in levels
    if {"high", "low"} <= levels:
        return value == "medium"
    return True


def shopper_type_consistent(shopper_type: dict) -> bool:
    values = shopper_type["value"]
    return len(set(values)) == len(values) and set(values) == {d["type"] for d in shopper_type["details"]}


def restrictions_consistent(restrictions: dict) -> bool:
    values = restrictions["value"]
    if UNKNOWN_RESTRICTION in values:
        return values == [UNKNOWN_RESTRICTION] and not restrictions["details"]
    return len(set(values)) == len(values) and set(values) == {d["restriction"] for d in restrictions["details"]}


def weak_price_or_deal_evidence(deal: dict, tier: dict) -> bool:
    tiers = Counter(detail["tier"] for detail in tier["details"])
    if tier["value"] in PRICE_TIERS and tiers and tiers[tier["value"]] < max(tiers.values()):
        return True
    if tier["value"] in ("premium", "budget") and tiers[tier["value"]] < 2:
        return True
    return deal["value"] == "high" and not any(d["signal_strength"] == "strong" for d in deal["details"])


def check_row(record: dict) -> CheckResult:
    reasons: set[str] = set()
    flags: set[str] = set()
    source = record.get("input") or {}
    interests = [interest for interest in source.get("commercial_interests") or [] if isinstance(interest, dict)]
    names = [interest.get("interest_name") for interest in interests]
    names += list(source.get("non_commercial_interest_names") or [])
    if not interests or not isinstance(source.get("life_stage"), dict) or has_empty_name(names):
        reasons.add("invalid_input")
    if has_duplicates([str(name or "") for name in names]):
        flags.add("duplicate_input_name")

    output = record.get("output")
    if not matches_format(output, OUTPUT_VALIDATOR, KEY_ORDERS):
        reasons.add("json_format")
        return reasons, flags

    deal, tier = output["deal_seeking"], output["price_tier"]
    shopping = output["affinity"]["shopping"]
    categories = [item["category"] for item in shopping["product_categories"]]
    if (not deal_seeking_consistent(deal)
            or not shopper_type_consistent(shopping["shopper_type"])
            or not restrictions_consistent(output["affinity"]["dining"]["restrictions"])
            or len(set(categories)) < len(categories)):
        reasons.add("inconsistent_content")
    if leaks_user_id(record):
        reasons.add("text_quality")

    if weak_price_or_deal_evidence(deal, tier):
        flags.add("weak_evidence")
    if not categories:
        flags.add("empty_product_categories")
    return reasons, flags


def main() -> None:
    args = parse_args("Rule-based cleaning of the Layer 4 commercial preference JSONL dataset.")
    report = clean(args, check_row, CATEGORIES, FLAG_CATEGORIES)
    print_summary(report, CATEGORIES)


if __name__ == "__main__":
    main()
