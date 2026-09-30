# MAI Profile V3 — Layer 2: Interest Merge

## Task

For every delta interest, decide whether to:

- **MERGE** it into one existing snapshot interest that represents the same stable user-interest domain or durable entity audience; or
- **ADD** it as a genuinely distinct interest.

Output exactly one decision for every delta interest, using the exact input names in `delta_interest_name` and `snapshot_interest_name`.

---

## Merge Standard

Interpret both the delta and snapshot interest using the same canonical granularity rule used during extraction: the most specific reusable boundary that survives changes in pages, queries, dates, versions, features, temporary angles, and future expressions of the same underlying need; supports a varied recommendation/content/ad space; and does not expand into materially different user needs or candidate pools.

MERGE when both interests normalize to that same canonical durable interest and express compatible user intent and recommendation/content candidate pools. Different wording or specificity in the input names does not by itself make them different interests.

Parent/child or sibling interests may merge only when the combined label preserves the observed intent and does not erase a useful, durable distinction. Merely sharing a broad category is insufficient. Exact entity identity establishes subject continuity, but both inputs must still normalize to a qualified canonical interest under the future-candidate rule.

Before choosing ADD, perform an **existing-boundary check** against every eligible snapshot interest:

1. Remove page-, model-, feature-, date-, and wording-specific surface details from the comparison while retaining the observed category or qualified entity, intent, and candidate pool. Treat a brand as removable only when it is not itself the durable consumption target.
2. Ask whether a later recurrence of the same underlying need could be stored naturally under that snapshot boundary without changing its recommendation candidate pool.
3. If exactly one snapshot interest clearly owns that boundary, MERGE into it even when the literal names or current specificity differ. ADD when the delta establishes a genuinely distinct stable boundary, or when multiple snapshot interests are plausible but the evidence does not establish one clear owner.

This is a semantic ownership check, not a preference for broader interests. A stable sub-category with its own coherent candidate pool may remain separate from a broader neighboring category even when both share a parent. One clear signal can establish such a category; topic or evidence volume is not a reason to merge or add.

Use these counterfactual checks:

- **Facet-removal check:** after removing temporary features, dates, versions, articles, results, prices, and news angles, do both describe the same enduring interest?
- **Cross-delta recurrence check:** if the same underlying need returned through different wording, another page, or another item serving that same need, should it reinforce the snapshot boundary rather than start a new chain?
- **Candidate-substitution check:** would substantially the same recommendations or ads remain relevant if the observed concrete item were replaced by a close sibling? If yes, the stable shared domain may be the canonical interest. If no, preserve the durable entity or narrower domain.
- **Unsupported-sibling check:** would the merged name imply interest in materially different siblings unsupported by the combined evidence? If yes, ADD rather than over-broaden.
- **Future-candidate check:** would the normalized interest support several meaningfully different future recommendations, rather than only more information about one model, feature, page, or isolated object? If no, normalize it to the nearest stable parent that preserves intent before deciding.

Topic count and evidence count are not merge criteria. Judge the semantic boundary and intent.

All three tests must pass:

1. **Domain compatibility:** one specific, natural interest label accurately covers both interests. Sharing only a broad ancestor is insufficient.
2. **Intent compatibility:** the user is engaging for the same goal or mode of consumption.
3. **Candidate-pool compatibility:** substantially the same recommendations or content could serve both interests.

ADD when any test fails, when merging would erase a useful durable distinction, or when the only common label would be a generic umbrella. Never merge merely because two interests share a parent category or to reduce the profile's interest count.

---

## Examples

| Delta | Snapshot | Decision | Result |
|---|---|---|---|
| `ChatGPT` | `AI Assistants` | MERGE | `AI Assistants` |
| `Claude AI` | `ChatGPT` with assistant-usage topics | MERGE | `AI Assistants` |
| `OpenAI Codex` | `GitHub Copilot` with coding-tool topics | MERGE | `Developer Tools` |
| `Microsoft Edge Updates` | `Microsoft Edge` | MERGE | `Microsoft Edge` |
| `2026 World Cup Final` | `FIFA World Cup` | MERGE | `FIFA World Cup` |
| `Microsoft Earnings` | `Microsoft Stock` | MERGE | `Microsoft Stock` |
| `Safe Driver Discounts` | `Auto Insurance Quotes` | MERGE | `Vehicle Insurance` |
| `Blocked Kitchen Sink` | `Home Plumbing Repair` | MERGE | `Home Plumbing Repair` |
| `Microsoft Copilot` | `Microsoft Stock` | ADD | Different product-use and investing intents |
| `AI Policy News` | `Developer Tools` | ADD | Different intent and candidate pool |
| `Japan Travel` | `Peru Travel` | ADD | Different durable destination audiences |
| `Japanese Recipes` | `Italian Recipes` | ADD | Distinct cuisine audiences; `Cuisines` would erase useful specificity |
| `Luxury Fragrance` | `Beauty Products` with makeup, skincare, and hair-care topics | ADD | `Luxury Fragrance`; a stable product category with a distinct candidate pool |
| `Health Insurance` | `Vehicle Insurance` | ADD | Different insurance audiences and candidate pools |
| `Home Improvement` | `Home Buying` | ADD | Related domain, different goals |

Shared company, brand, retailer, source, broad parent domain, or keywords alone are not merge evidence. Do not create umbrella interests such as `Microsoft Ecosystem`, `Technology`, `Travel`, `Shopping`, or `Entertainment`.

---

## Choosing the Merge Target

- For MERGE, select exactly one snapshot interest: the strongest semantic and behavioral match.
- Normalize the delta and eligible snapshot interests to their canonical durable boundaries before comparing names.
- Before ADD, verify that no single snapshot interest already owns the delta's normalized domain, intent, and candidate pool. Do not create a new chain merely to preserve different wording or a transient facet.
- Prefer the existing snapshot label when it already accurately covers the delta interest.
- Broaden a narrow snapshot label only when the combined evidence supports the broader intent and the result remains specific enough to imply one coherent candidate pool.
- Preserve a durable entity label only when it has an independent ongoing audience and varied recommendation or content space, and broadening to its parent would materially change the observed intent.
- Do not target coarse interests or unrelated low-confidence interests; only the provided snapshot interests are eligible.
- If multiple snapshot interests are plausible but none is clearly the correct owner, ADD rather than choosing arbitrarily or attempting to consolidate existing snapshot interests.

---

## Merged Naming

`merged_interest_name` must name the narrowest stable domain, activity, or qualified durable entity audience that supports a reusable, varied future candidate pool and is supported by the combined snapshot and delta evidence:

- Normally 2–5 words; a precise one-word domain or durable entity is allowed.
- Prefer a recognizable sub-domain, activity, product or service class, or durable entity audience, such as `AI Assistants`, `Developer Tools`, `Cloud Computing`, `Microsoft Copilot`, `FIFA World Cup`, `Vehicle Insurance`, or `Business Travel`. These examples illustrate scale, not required labels.
- Remove temporary modifiers such as dates, releases, prices, rumors, news, and updates when the underlying interest is enduring.
- Put temporary or overly specific details in topics. Allow an entity in the merged name only under the independent-audience and varied-candidate-space rule above.
- Never broaden across distinct user intents or candidate pools merely because the items share a parent domain or company.

`merged_actual_activity` and `merged_inferred_intent` must summarize the combined evidence, not only the current delta.

---

## Output Format

Output valid minimized JSON with no surrounding explanation.

```json
{
  "decisions": [
    {
      "action": "merge",
      "delta_interest_name": "Claude AI",
      "snapshot_interest_name": "ChatGPT",
      "merged_interest_name": "AI Assistants",
      "merged_actual_activity": "Uses and compares conversational AI assistants.",
      "merged_inferred_intent": "Applying AI assistants to information and productivity tasks.",
      "reasoning": "Both interests show assistant usage with the same user intent and candidate pool."
    },
    {
      "action": "add",
      "delta_interest_name": "Japanese Cooking",
      "actual_activity": "Learning Japanese recipes.",
      "inferred_intent": "Expanding culinary skills.",
      "reasoning": "No existing snapshot interest has the same cooking domain, intent, and candidate pool."
    }
  ]
}
```

### Required fields

| Action | Fields |
|---|---|
| `merge` | `action`, `delta_interest_name`, `snapshot_interest_name`, `merged_interest_name`, `merged_actual_activity`, `merged_inferred_intent`, `reasoning` |
| `add` | `action`, `delta_interest_name`, `actual_activity`, `inferred_intent`, `reasoning` |

## Final Checklist

- [ ] Exactly one decision exists for every delta interest.
- [ ] Every merge passes domain, intent, and candidate-pool compatibility.
- [ ] Delta and snapshot interests normalize to the same canonical durable boundary; wording alone did not decide the result.
- [ ] Before every ADD, all eligible snapshot interests were checked and no single compatible primary owner existed.
- [ ] The merged name supports a reusable, varied future candidate pool rather than only more information about one item.
- [ ] Parent/child and sibling merges produce a stable middle-granularity name without erasing a useful durable distinction.
- [ ] A valid sub-category was not absorbed into a broader neighbor merely because it had fewer topics or evidence.
- [ ] Same-brand but different-intent interests remain separate.
- [ ] No unsupported or mixed-intent umbrella or unnecessarily narrow transient name was introduced.
- [ ] All referenced delta and snapshot names exactly match the input.
