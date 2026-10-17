# V2 User Profile Dataset — Schema Guide

**Dataset:** [`yufan/user_profile_dataset`](https://huggingface.co/datasets/yufan/user_profile_dataset) · **Configs:** the 8 `V2_*` configs · **Teacher:** GPT-5.6

For every config this guide shows what the model reads (**Input**) and what it must return (**Output**).

---

## 1. Overview

| Config | Task | Train | Test |
|---|---|---:|---:|
| `V2_User_Profile_L1_gpt56` | Raw activities → interests | 267,810 | 5,378 |
| `V2_User_Profile_L1_Negative_gpt56` | Negative feedback → rejected topics | 557,734 | 11,184 |
| `V2_User_Profile_L2_gpt56` | Existing profile + new interests → merge / add decisions | 259,918 | 5,199 |
| `V2_User_Profile_L3_Commercial_gpt56` | Interests → commercial label, brands, search queries | 145,277 | 7,059 |
| `V2_User_Profile_L3_Persona_gpt56` | Interests → persona + category | 77,312 | 3,780 |
| `V2_User_Profile_L4_Biography_gpt56` | Interests + facts → life stage, attributes, biography | 161,793 | 8,103 |
| `V2_User_Profile_L4_CommercialPreference_gpt56` | Commercial interests → shopping preferences | 141,250 | 7,093 |
| `V2_User_Profile_L4_Hyper_gpt56` | Commercial interests + context → ad missions + new queries | 94,514 | 4,843 |

Which input fields can be empty, and whether "empty" means a missing key, `{}`, `[]`, or `""`, is measured per config
in the [Appendix](#appendix--input-completeness-audit-empty-vs-missing).

---

## 2. Row format (all configs)

```ts
{
  uid: string,             // 16-char hex user ID (L1 Negative: hash of the row's actions, the data has no user ID)
  messages: [
    { role: "user",      content: PROMPT + "\n\nInput:\n" + INPUT },
    { role: "assistant", content: OUTPUT }          // minimized JSON
  ],
  source: "GPT56",
  domain: string,          // see the table below
  think_type: "none",
  language?: string        // see the table below
}
```

- One fixed prompt per config, no system message, always one user turn and one assistant turn.
- `OUTPUT` always parses with `json.loads`.

How to read a row:

```python
import json

prompt, _, raw_input = row["messages"][0]["content"].rpartition("\nInput:\n")
output = json.loads(row["messages"][1]["content"])
if row["domain"] in ("maiprofile-layer1-combined", "maiprofile-layer1-negative"):  # JSONL: one object per line
    input_data = [json.loads(line) for line in raw_input.split("\n")]
else:
    input_data = json.loads(raw_input)
```

| Config | `domain` | `language` column | Input format |
|---|---|---|---|
| L1 | `maiprofile-layer1-combined` | content language (= output `predicted_content_locale`) | JSONL |
| L1 Negative | `maiprofile-layer1-negative` | market: `en` or `glb` | JSONL |
| L2 | `maiprofile-layer2-combined` | *(no column)* | JSON |
| L3 Commercial | `maiprofile-layer3-commercial` | input `query_language` | JSON |
| L3 Persona | `maiprofile-layer3-persona` | always `en` | JSON |
| L4 Biography | `maiprofile-layer4-biography` | always `en` | JSON |
| L4 Commercial Preference | `maiprofile-layer4-commercial-preference` | always `en` | JSON |
| L4 Hyper | `maiprofile-layer4-hyper` | input `query_language` | JSON |

---

## 3. How the layers connect

```
L1  raw activities -> interests
 `-- L2  merge new interests into the user's profile
      |-- L3 Persona       persona + category for each interest
      `-- L3 Commercial    commercial label, brands, queries for each interest
           `-- L4 Biography                  life stage, user attributes, biography
                `-- L4 Commercial Preference  shopping preferences
                     `-- L4 Hyper              ad missions + new queries

L1 Negative  negative feedback -> rejected topics   (standalone: no user ID to join with the other layers)
```

---

## 4. Shared types and notation

```ts
// Source: activity source. The 9 values after "Shopping" are rare (259 occurrences, only in L3 Commercial / L4
// Biography inputs; "Home Depot" 6)
type Source = "MSN" | "Edge" | "Uet" | "Bing" | "Ads" | "ChromeImports" | "Shopping" | "Linkedin"
            | "YouTube" | "Amazon" | "Xbox" | "Facebook" | "Google Play" | "Walmart" | "Yahoo" | "Home Depot";
type Lang   = "en" | "de" | "fr" | "es" | "ja" | "pt" | "nl" | "it" | "sv" | "pl" | "zh-Hans" | "zh-Hant";
type Level  = "high" | "medium" | "low";
type Scale  = "Low" | "Medium-Low" | "Medium" | "Medium-High" | "High";
type Signal = "strong" | "moderate" | "weak";
```

- `field?` = optional field; `~N%` = how often it is present.
- All counts (`1–39, median 10`) and percentages are measured on every row of the config (train + test).

---

## L1 — Interest Extraction

`V2_User_Profile_L1_gpt56` · Find the user's interests in one window of activities.

**Input** — JSONL, one activity per line (1–200 lines, median 37)

```ts
{ idx: int, date: "YYYY-MM-DD", source: Source, action: string }   // action: page title, query, or article title
```

**Output**

```ts
{
  predicted_content_locale: Lang | "mix",   // main language of the activities
  interests: [{                             // 0–39, median 10 (0.3% of rows have none)
    interest_name: string,
    topics: [{                              // 1–53, median 2
      topic: string,
      source: Source[],                     // 1–6
      evidence: int[]                       // input idx
    }],
    actual_activity: string,                // one factual sentence
    inferred_intent: string                 // the intent one step above the activity
  }]
}
```

---

## L1 Negative — Rejected Topics

`V2_User_Profile_L1_Negative_gpt56` · Find the topics the user rejected in one batch of negative feedback.

**Input** — JSONL, one signal per line (1–148 lines, median 1; 74% of rows have one signal; no dates)

```ts
{
  idx: int,                       // 0..n-1
  source: "MSN" | "Ads",
  detailed_source: "ContentUserFeedback" | "AdUserFeedback",
  action: string                  // MSN: "Downvote | <content title>"
                                  // Ads: "NotInterested | <advertiser> | <ad title>"
}
```

**Output**

```ts
{
  negative_interests: [{          // 1–103, median 1
    keywords: string,             // the rejected topic: 1–8 lowercase words (median 4), any language
    evidence: int[],              // input idx it is based on (95% cite one signal)
    source: ("MSN" | "Ads")[],    // sources of the cited signals
    detailed_source: string,      // detailed source of the cited signals
    signal_type: string[],        // lowercase action prefixes: "downvote" | "notinterested"
    reliability_score: float      // 0–1 (observed 0.72–0.99, median 0.96)
  }]
}
```

**Notes**

- Every input signal is cited by at least one interest.
- `MSN` always pairs with `ContentUserFeedback` and `downvote`; `Ads` with `AdUserFeedback` and `notinterested`
  (one interest has the combined `signal_type` `"repeatedad~inappropriatead~notinterested"`).
- `keywords` names a topic, never a reaction such as "offensive"; 94% are plain ASCII (mostly English).

---

## L2 — Profile Merge

`V2_User_Profile_L2_gpt56` · Merge each new interest into the user's existing profile, or add it as new.

**Input**

```ts
{
  // existing profile, 0–275 interests (median 39)
  snapshot: [{ interest_name: string, actual_activity: string, topics: string[] }],
  // new interests, 1–95 (median 10)
  delta:    [{ interest_name: string, actual_activity: string, topics: string[] }]
}
```

**Output** — one decision per `delta` interest, in the same order (66 rows, 0.02%, list them in another order)

```ts
{
  decisions: [
      { action: "merge",                     // ~60%
        delta_interest_name: string, snapshot_interest_name: string,
        merged_interest_name: string, merged_actual_activity: string, merged_inferred_intent: string }
    | { action: "add",                       // ~40%
        delta_interest_name: string, actual_activity: string, inferred_intent: string }
  ]
}
```

---

## L3 Commercial — Commercial Classification

`V2_User_Profile_L3_Commercial_gpt56` · Decide whether advertisers could target each interest, and list its entities
and search queries.

**Input**

```ts
{
  interests: [{                              // 1–30, median 24
    interest_name: string,
    actual_activity: string,
    sources: Source[],
    topics: [{ topic: string, actions: string[] }]   // actions: raw activity texts (0–3)
  }],
  query_language: Lang
}
```

**Output** — one entry per input interest

```ts
{
  interest_commercial: [{
    interest_name: string,
    commercial: boolean,                                       // true ~75%
    commercial_score: "high" | "medium" | "low" | null,
    intent_funnel_stage: "discovery" | "research" | "consideration" | "purchase" | "post-purchase" | null,
    brands: string[],
    retailers: string[],
    products: string[],
    predicted_queries: string[]                                // ≤ 3, in query_language
  }]
}
```

**Notes**

- When `commercial` is `false`: `commercial_score` and `intent_funnel_stage` are `null`, and all lists are `[]`.

---

## L3 Persona — Persona & Category

`V2_User_Profile_L3_Persona_gpt56` · Write a persona and a category path for each interest.

**Input**

```ts
{
  interests: [{                 // 1–229, median 42
    interest_name: string,
    actual_activity: string,
    inferred_intent: string,
    topics: string[]
  }]
}
```

**Output** — one entry per input interest

```ts
{
  interest_personas: [{
    interest_name: string,
    category: string,           // "/" path, broad → specific,
                                //   e.g. "/Technology/Artificial Intelligence/AI Assistants"
    persona: string
  }]
}
```

---

## L4 Biography — Life Stage, Attributes, Biography

`V2_User_Profile_L4_Biography_gpt56` · Infer the user's life stage and attributes, and write a short biography.

169,896 rows, 20,861 users (19,873 train / 988 test, no overlap). The GPT-5.6 `user_attributes.age_group.signals`
field has been removed from the output (it was in 115,429 rows, 67.9%: every row whose `age_group` was not declared).
Nothing else in the output was changed.

**Input**

```ts
{
  declared_attributes: {                     // {} in 65% of rows
    age_group?: "18 - 24" | "25 - 34" | "35 - 49" | "50 - 65" | "65+",   // ~32%
    gender?: "Male" | "Female"                                           // ~26%
  },
  facts: {                                   // {} in 55% of rows
    market?: string,                         // ~28%, e.g. "en-US"
    location_data?: {                        // ~30%
      current_location: string,
      place_frequency:  [{ place: string,  count: int }],
      region_frequency?: [{ region: string, count: int }],   // ~97% of location_data
      recent_visits?: string[]                               // ~53% of location_data
    },
    linkedin_profile?: {                     // ~16%; sparse fields, each a string or string[]
      profile_geo?: string,                  // 99.9% of linkedin_profile
      industries?, degree?, degrees?, fieldOfStudy?, areas?, titles?, current_roles?,
      yearsOfExperience?, experience_band?, seniorities?, functions?, companies?, current_company?
    }
  },
  interests: [{                              // 1–203, median 41
    interest_name: string,
    actual_activity: string,
    inferred_intent: string,
    persona: string,                         // from L3 Persona
    confidence_score: float,
    count: int,
    first_detect_date: "YYYY-MM-DD",
    last_detect_date: "YYYY-MM-DD",
    source: Source[],
    topics?: string[],                       // missing for 4 interests (4 rows)
    predicted_queries?: string[],            // ~68%, from L3 Commercial
    products?: string[],                     // ~51%
    brands?: string[]                        // ~44%
  }]
}
```

**Output**

```ts
type AttrSource = "declared" | "behavioral_inference" | "location_data" | "linkedin" | "unknown";
type Attr = { evidence: string[], value: string, confidence: Level, source: AttrSource };

{
  life_stage: {
    value: "single" | "married" | "parenting" | "caregiving" | "job_seeking" | "new_grad" | "retirement"
         | "unknown",
    confidence: Level,
    evidence: string[]                       // 1–3
  },
  user_attributes: {                         // always all 8 keys, in this order
    age_group: Attr,                         // "13 - 17" | "18 - 24" | "25 - 34" | "35 - 49" | "50 - 65" | "65+"
                                             //   | "Unknown"
    gender: Attr & { lean: { male: string[], female: string[] } },   // "Male" | "Female" | "Unknown"
    estimated_income_level: Attr,            // "High" | "Medium" | "Low" | "Unknown"
    home_location: Attr,                     // place name | "Unknown"
    occupation: Attr,                        // occupation category | "Unknown"
    company: Attr,                           // company name | "Self-Employed" | "Unknown"
    career_stage: Attr,                      // "Student" | "Entry Level" | "Mid-Level Professional"
                                             //   | "Senior Professional" | "Manager" | "Director/Executive"
                                             //   | "Business Owner/Self-Employed" | "Retired"
                                             //   | "Not in Workforce" | "Unknown"
    education_area: Attr                     // field of study | "Unknown"
  },
  biography: string                          // 2–3 sentences, 143–550 chars (median 323)
}
```

Key order is fixed: `Attr` is always `{evidence, value, confidence, source}`, and `gender` is
`{lean, evidence, value, confidence, source}`.

**Output statistics** (169,896 rows)

`life_stage`:

| `value` | Rows | Share |
|---|---:|---:|
| `unknown` | 90,953 | 53.5% |
| `retirement` | 27,274 | 16.1% |
| `parenting` | 22,967 | 13.5% |
| `job_seeking` | 17,973 | 10.6% |
| `single` | 7,078 | 4.2% |
| `caregiving` | 2,397 | 1.4% |
| `married` | 1,131 | 0.7% |
| `new_grad` | 123 | 0.1% |

`confidence`: low 77.1% / medium 18.7% / high 4.3%. `evidence` items: 1 19.9% / 2 60.8% / 3 19.3%.

`user_attributes` (all shares are of all rows):

| Attribute | `Unknown` | `source` (non-unknown) | `confidence` low / medium / high | Most common known values |
|---|---:|---|---|---|
| `age_group` | 5.6% | behavioral_inference 61.9%, declared 32.1%, linkedin 0.5% | 58.5 / 5.7 / 35.8% | `35 - 49` 50.5%, `50 - 65` 21.9%, `65+` 18.3%, `25 - 34` 2.5%, `18 - 24` 1.2%, `13 - 17` 77 rows |
| `gender` | 2.3% | behavioral_inference 71.5%, declared 26.1%, linkedin 3 rows | 37.3 / 36.6 / 26.1% | `Male` 59.0%, `Female` 38.6% |
| `estimated_income_level` | 2.5% | behavioral_inference 97.4%, linkedin 0.1% | 99.4 / 0.6 / 42 rows | `Medium` 96.4%, `High` 1.0%, `Low` 0.1% |
| `home_location` | 31.4% | behavioral_inference 30.4%, location_data 29.9%, linkedin 8.2% | 42.4 / 32.2 / 25.4% | 10,713 distinct; `United States` 1.2%, `France` 1.0%, `United Kingdom` 0.9% |
| `occupation` | 66.1% | behavioral_inference 25.3%, linkedin 8.6% | 89.8 / 1.1 / 9.1% | 8,475 distinct; `Business / Finance Professional` 4.4%, `Technology Professional` 1.9% |
| `company` | 95.9% | linkedin 3.3%, behavioral_inference 0.8% | 96.5 / 0.3 / 3.2% | 905 distinct; `Self-Employed` 0.5%, then ≤ 58 rows each |
| `career_stage` | 57.2% | behavioral_inference 34.1%, linkedin 8.7% | 86.7 / 4.3 / 8.9% | `Senior Professional` 24.0%, `Retired` 10.5%, `Entry Level` 2.7%, `Director/Executive` 1.5% |
| `education_area` | 74.3% | behavioral_inference 20.3%, linkedin 5.4% | 94.4 / 0.2 / 5.5% | 5,046 distinct; `Business / Accounting` 4.5%, `Health Sciences` 1.7% |

**Notes**

- `value` is `"Unknown"` ⇔ `source` is `"unknown"` ⇔ `evidence` is `[]`, for every attribute in every row. Known
  values always have 1–3 evidence items.
- Declared attributes are copied verbatim: all 54,467 rows with a declared `age_group` output that same value with
  `source: "declared"` (the same holds for the 44,399 declared genders).
- `gender.lean` is always `{male, female}` (0–10 entries each). When gender is inferred, every entry is
  `"<interest name> (strong|moderate|weak)"`, e.g. `"Football (moderate)"` (weak 51.0%, moderate 44.0%, strong 4.9%
  of entries). When gender is declared, `lean` is not a scored list: it holds one free-text entry such as
  `"Declared male"` (43,260 rows) or is empty in both lists (1,139 rows).
- `age_group` has no `signals` field. The evidence text was not edited, so 25,076 rows (14.8%) still mention
  "later-life" (e.g. `"... provide two later-life clusters"`).

---

## L4 Commercial Preference — Shopping Preferences

`V2_User_Profile_L4_CommercialPreference_gpt56` · Summarize how the user shops.

**Input**

```ts
{
  life_stage: { value: string, confidence: Level, evidence: string[] },   // L4 Biography output
  commercial_interests: [{                   // 1–119, median 27
    interest_name: string,
    persona: string,
    brands: string[],
    retailers: string[],
    products: string[],
    topics: [{ topic: string, evidence: string[] }]   // evidence: raw activity texts
  }],
  non_commercial_interest_names: string[]
}
```

**Output**

```ts
type Detail<K extends string, V> = { area: string, evidence: string, signal_strength: Signal } & { [key in K]: V };

{
  discount_sensitivity:  { value: Scale | "Unknown", details: Detail<"sensitivity", Scale>[] },
  price_sensitivity:     { value: Scale | "Unknown", details: Detail<"sensitivity", Scale>[] },
  subscription_affinity: { value: Scale | "Unknown", details: Detail<"affinity", Scale>[] },
  affinity: {
    shopping: {
      product_categories: [{ category: AdsVertical, description: string }],
      shopper_type: { value: ShopperType[], details: Detail<"type", ShopperType>[] }
    },
    dining: {
      restrictions: { value: Diet[], details: Detail<"restriction", Diet>[] }   // value ["Unknown"] when no signal
    }
  }
}

type AdsVertical = "HomeGarden" | "ComputersConsumerElectronics" | "RetailersGeneralMerchandise" | "TravelTourism"
                 | "Finance" | "Apparel" | "Vehicles" | "InternetTelecom" | "Health" | "BeautyPersonalCare"
                 | "SportsFitness";          // Ads Vertical L1
type ShopperType = "ResearchDriven" | "DealDriven" | "BrandRetailerLoyal" | "ValueDriven"
                 | "NoveltySeeker" | "ConvenienceDriven" | "PremiumDriven"
                 | "ImpulseDriven";          // ImpulseDriven: 1 row
type Diet        = "Unknown" | "Vegan" | "Vegetarian" | "No Gluten" | "Low-Carb" | "Keto" | "Halal" | "Dairy-Free"
                 | "Kosher" | "Pescatarian" | "Paleo" | "No Nuts"
                 | "No Shellfish";           // the last 5 are rare (166 occurrences; No Shellfish 2)
```

**Notes**

- When a `value` is `"Unknown"`, its `details` is `[]`.

---

## L4 Hyper — Ad Missions & Queries

`V2_User_Profile_L4_Hyper_gpt56` · Turn the user's commercial interests into missions for ads retrieval, each with new
search queries.

**Input** — a JSON object whose first four fields are **strings**

```ts
{
  personal_context: string,        // Markdown list or "": Market, Current location, Age group, Gender,
                                   //   Recent visits
  professional_context: string,    // Markdown list, "" in ~88%: Current company, Current roles, Functions,
                                   //   Industries, Seniorities, Years of experience, Degrees, Education areas
  commercial_preferences: string,  // a JSON string: parse it again (format below)
  commercial_interests: string,    // Markdown, one block per interest (format below)
  query_language: Lang
}
```

`commercial_preferences`, after parsing — a short form of the L4 Commercial Preference output (`details` replaced by
`confidence_score`):

```ts
{
  discount_sensitivity?:  { value: Scale, confidence_score: float },
  price_sensitivity?:     { value: Scale, confidence_score: float },
  subscription_affinity?: { value: Scale, confidence_score: float },
  affinity?: {                                   // missing in 2 rows
    shopping: {
      product_categories?: [{ category: string, description: string }],   // missing in 4 rows
      shopper_type?: { value: ShopperType[], confidence_score: float }
    },
    // Diet plus "Low-Purine" / "Low-Sodium" (1 row each)
    dining?: { restrictions: { value: Diet[], confidence_score: float } }
  }
}
```

`commercial_interests` block (in < 0.1% of rows a block misses its Category, Persona, Activity, or Intent line):

```md
### Interest 1: <interest_name>
- Category: /A/B/C
- Persona: ...
- Activity: ...
- Intent: ...
- Existing queries:
  - <query>
- Known entities:          (optional)
  - Brands: a, b
  - Retailers: a, b
  - Products: a, b
```

**Output**

```ts
{
  missions: [{                                   // 0–17, median 12 (128 rows have none)
    mission_name: string,                        // in query_language
    source_interests: string[],                  // exact input interest names
    scenarios: ("shopping" | "travel" | "technology" | "hobbies" | "learning" | "fitness" | "dining")[],
    predicted_brands: string[],                  // ≤ 3 new brands
    predicted_queries: string[]                  // 1–4 queries, each 1–8 words (median 4), in query_language
  }]
}
```

**Output statistics** (99,357 rows, 1,016,271 missions, 2,213,233 queries)

- Missions per row: 0–17, mean 10.2, median 12. Queries per mission: 1–4, mean 2.18, median 2.
- `scenarios` (a mission can have several; share of missions): shopping 81.3%, travel 19.2%, technology 15.5%,
  hobbies 12.1%, learning 6.6%, fitness 3.7%, dining 2.9%.
- `predicted_brands` per mission: 0 74.4%, 1 16.6%, 2 5.7%, 3 3.2%.
- Full chat length (Qwen3.5-4B chat template, thinking disabled): mean 6,414, P50 5,879, P90 11,846, P99 15,026,
  max 17,511 tokens.

**Notes**

- Each query is now a plain string. The per-query `value_type`, `source_query_refs`, `delta_source`,
  `delta_evidence`, and `decision_change` fields and the per-mission `enrichment_sources` field have been removed
  from every row (2,213,233 queries in 1,016,271 missions). This cuts 74% of the output characters; the
  full chat length drops from a mean of 8,137 to 6,414 tokens.
- The prompt was **not** changed: it still asks the model to label, cite, and source each query, and its JSON template
  still shows the removed fields, so the prompt and the output no longer match.

---

## Appendix — Input completeness audit (empty vs. missing)

Every row of all 47 V2 files (1,758,247 rows, train + test) was scanned, one file at a time, from L4 back to L1. For
each input path we counted whether the **key is missing** or the key is present with an **empty value** (`{}`, `[]`,
`""`). L4 Hyper's JSON-string and Markdown fields were parsed again and counted line by line.

- `null` never appears in any V2 input.
- Every row parses, has exactly one `user` + one `assistant` turn, and uses the single prompt of its config.
- "rare" below = < 0.5% of the parent items.

### A.1 Summary — how each config says "no data"

| Config | Rows | Top-level keys (always all present) | Top-level value can be empty | Absent optional data inside |
|---|---:|---|---|---|
| L4 Hyper | 99,357 | `personal_context`, `professional_context`, `commercial_preferences`, `commercial_interests`, `query_language` | `personal_context` = `""` 44.3%; `professional_context` = `""` 87.6% | **line / key omitted** inside the Markdown and JSON strings (never an empty value) |
| L4 Commercial Preference | 148,343 | `life_stage`, `commercial_interests`, `non_commercial_interest_names` | `non_commercial_interest_names` = `[]` 15.4% | **key kept, value `[]`** (no key is ever missing) |
| L4 Biography | 169,896 | `declared_attributes`, `facts`, `interests` | `declared_attributes` = `{}` 64.9%; `facts` = `{}` 54.9% | **key omitted** (never `[]` / `""` / `{}` below the top level) |
| L3 Commercial | 152,336 | `interests`, `query_language` | no | key kept; rare `""` / `[]` |
| L3 Persona | 81,092 | `interests` | no | key kept; rare `""` / `[]` |
| L2 | 265,117 | `snapshot`, `delta` | `snapshot` = `[]` in 24 rows | key kept; rare `""` / `[]` |
| L1 Negative | 568,918 | JSONL lines `{idx, source, detailed_source, action}` | no | every line has all 4 keys, none empty |
| L1 | 273,188 | JSONL lines `{idx, date, source, action}` | no | every line has all 4 keys, none empty |

L1–L3 inputs are uniform. Only L4 varies, and each L4 config uses its own convention — even for the same data. For
example, an interest with no brands is:

| Config | Representation | Share of interests |
|---|---|---:|
| L4 Biography | `brands` key missing | 55.5% |
| L4 Commercial Preference | `"brands": []` | 34.4% |
| L4 Hyper | no `- Brands:` line (or no `- Known entities:` block at all) | 16.0% no block; 21.6% of blocks lack Brands |

### A.2 L4 Hyper

**Row situations** — the two context strings are either `""` or a Markdown list; they are never missing:

| `personal_context` | `professional_context` | Rows | Share |
|---|---|---:|---:|
| `""` | `""` | 43,996 | 44.28% |
| non-empty | `""` | 42,998 | 43.28% |
| non-empty | non-empty | 12,357 | 12.44% |
| `""` | non-empty | 6 | 0.01% |

**`commercial_preferences`** (JSON string; never `""`) — top-level keys after parsing:

| Keys | Rows | Share |
|---|---:|---:|
| `affinity`, `discount_sensitivity`, `price_sensitivity` | 36,836 | 37.07% |
| `affinity`, `discount_sensitivity`, `price_sensitivity`, `subscription_affinity` | 25,804 | 25.97% |
| `affinity` only | 18,260 | 18.38% |
| `affinity`, `price_sensitivity` | 10,342 | 10.41% |
| `affinity`, `discount_sensitivity` | 4,543 | 4.57% |
| `affinity`, `price_sensitivity`, `subscription_affinity` | 2,112 | 2.13% |
| `affinity`, `subscription_affinity` | 821 | 0.83% |
| `affinity`, `discount_sensitivity`, `subscription_affinity` | 637 | 0.64% |
| `price_sensitivity` only (no `affinity`) | 2 | 0.00% |

- A missing sensitivity / affinity key most likely means that value was `"Unknown"` (or `[]`) in the L4 Commercial
  Preference output: the shares are close (`discount_sensitivity` missing 31.7% vs. `Unknown` 32.7% in the CP output,
  `price_sensitivity` 24.4% vs. 27.4%) but not identical (`subscription_affinity` 70.4% vs. 61.2%), so it was not
  joined row by row.
- `affinity.shopping` is always present; `shopping.product_categories` is missing in 4 rows, `shopping.shopper_type`
  in 7.6%. `affinity.dining` is present in only 1.4% (1,401 rows).
- No empty values: every present list has ≥ 1 item.

**`personal_context`** labels (when non-empty, 55,355 rows):

| Label | Present | Notes |
|---|---:|---|
| `Gender` | 85.5% | `M` / `F` |
| `Current location` | 82.8% | free text, mixed formats (`Sydney, New South Wales, AU`, `united states`) |
| `Age group` | 77.9% | **not an age band**: raw age `18`–`120` (90%) or a letter code `S` / `R` / `T` / `P` / `Q` (10%) |
| `Market` | 63.8% | `en-US` / `en-GB` / `en-CA` / `en-AU` / `en-IN` |
| `Recent visits` | 34.1% | comma-joined places |

Most common label sets: {Age group, Current location, Gender, Market} 22.3%; + Recent visits 20.8%; {Age group,
Gender, Market} 12.7%; {Current location} only 12.4%; 22 combinations in total.

**`professional_context`** labels (when non-empty, 12,363 rows; 54 combinations): `Industries` 96.6%, `Current roles`
84.5%, `Seniorities` 84.0%, `Functions` 79.0%, `Years of experience` 73.2%, `Degrees` 53.8%, `Education areas` 53.3%,
`Current company` 31.6%.

**`commercial_interests`** blocks (3,497,853 blocks, 1–119 per row, median 31):

| Line | Present |
|---|---:|
| `### Interest N: <name>`, `Existing queries` (1–3 queries) | always |
| `Category`, `Persona`, `Activity`, `Intent` | ≥ 99.98% each (missing 509 / 491 / 636 / 141 blocks) |
| `Known entities` | 84.0% |
| ↳ `Brands` / `Products` / `Retailers` (within Known entities) | 78.4% / 90.5% / 62.1% |

A sub-line is never written with an empty value: when there is no data, the line is left out.

### A.3 L4 Commercial Preference

Every key is always present — absence is always `[]` or `""`:

| Path | Empty value |
|---|---|
| `non_commercial_interest_names` | `[]` in 22,878 rows (15.42%) |
| `commercial_interests[].brands` | `[]` 34.36% of interests |
| `commercial_interests[].products` | `[]` 23.80% |
| `commercial_interests[].retailers` | `[]` 47.43% |
| `commercial_interests[].persona` | `""` 3,414 (0.08%) |
| `commercial_interests[].topics` | `[]` 2 interests |
| `commercial_interests[].topics[].evidence` | `[]` 627 topics |
| `commercial_interests[].topics[].topic` | `""` 54 topics |

- `commercial_interests` is never `[]` (1–119 items).
- `life_stage` always has `value`, `confidence`, `evidence` (1–3). "No life stage" is the value `"unknown"`
  (55.7%), not a missing key. 2 rows have the off-list value `"Student"`.

### A.4 L4 Biography

`declared_attributes` and `facts` are always present: `{}` when there is nothing, otherwise only the known sub-keys
appear. Row situations (`declared_attributes` × `facts`):

| `facts` \ `declared_attributes` | `{}` | `{age_group, gender}` | `{age_group}` | `{gender}` | Total |
|---|---:|---:|---:|---:|---:|
| `{}` | 91,920 | 712 | 528 | 149 | 93,309 |
| `{location_data, market}` | 4,252 | 14,680 | 6,308 | 1,494 | 26,734 |
| `{market}` | 1,654 | 6,001 | 2,713 | 1,412 | 11,780 |
| `{linkedin_profile}` | 10,187 | 744 | 235 | 113 | 11,279 |
| `{location_data}` | 1,464 | 5,823 | 2,998 | 854 | 11,139 |
| `{linkedin_profile, location_data, market}` | 437 | 5,013 | 1,253 | 506 | 7,209 |
| `{linkedin_profile, location_data}` | 219 | 4,421 | 793 | 362 | 5,795 |
| `{linkedin_profile, market}` | 113 | 1,822 | 423 | 293 | 2,651 |
| **Total** | 110,246 | 39,216 | 15,251 | 5,183 | 169,896 |

- 54.1% of rows are "behavior only": both `{}`.
- `facts.location_data`: `current_location` and `place_frequency` always; `region_frequency` missing in 2.6%;
  `recent_visits` missing in 47.0%. No empty lists.
- `facts.linkedin_profile`: 59 key combinations; 36.4% contain only `profile_geo`. Value type depends on the field:
  `profile_geo`, `current_company`, `experience_band` are always `string`; `current_roles`, `areas`, `degrees` are
  always `string[]`; `industries`, `titles`, `functions`, `degree`, `seniorities`, `fieldOfStudy`, `companies`,
  `yearsOfExperience` are `string` when there is one value and `string[]` (≥ 2) when there are more.
- `interests[]`: `predicted_queries` / `products` / `brands` are **missing** (never `[]`) for non-commercial interests.
  Key sets: all three 39.6%; none 32.2%; `predicted_queries` + `products` 11.9%; `predicted_queries` only 11.4%;
  `predicted_queries` + `brands` 4.9%. `topics` is missing for 4 interests.
- Rare empty values: `interests[].persona` `""` 9,135; `actual_activity` `""` 1,679; `inferred_intent` `""` 684;
  `source` `[]` 166.

### A.5 L3, L2, L1

No missing keys anywhere; each config has exactly one row situation (L2: two). Only rare empty values:

| Config | Empty values |
|---|---|
| L3 Commercial | `interests[].actual_activity` `""` 706; `sources` `[]` 79; `topics` `[]` 3; `topics[].actions` `[]` 638 |
| L3 Persona | `interests[].actual_activity` `""` 740; `inferred_intent` `""` 176; `topics` `[]` 2; `topics[]` `""` 55 |
| L2 | `snapshot` `[]` 24 rows; `delta[].actual_activity` `""` 10,356 (0.31%); `snapshot[].actual_activity` `""` 3,384; `topics` `[]` 16; `topics[]` `""` 292 |
| L1 Negative | none. Row sources: MSN only 93.9%, Ads only 5.3%, both 0.8% |
| L1 | none. 109 source mixes; the most common are `Uet` only 16.2% and `Ads+Bing+MSN+Uet` 12.1% |
