# V3 User Profile Dataset: Input and Output Schemas

This guide covers the 8 `V3_*` configs in
[`yufan/user_profile_dataset`](https://huggingface.co/datasets/yufan/user_profile_dataset).
The 7 profile tasks L1–L4 use the GPT-5.6 Final teacher data (`Training_Data_InferredByGPT56_Final/process_dataset`).
`V3_User_Profile_L1_Negative_gpt56` copies `V2_User_Profile_L1_Negative_gpt56` (train unchanged, test capped at
5,000 rows), because the Final data has no negative-feedback calls. There is no V3 Ranker config;
`V2_User_Profile_Ranker` stays as it was.

Each task shows what the model reads and what it returns. The notes explain which fields are optional, how
missing data is represented, and what changed from V2. Section 5 lists the V2 → V3 changes in one place.

## 1. Dataset overview

| Config | Folder | Task | Train | Test |
|---|---|---|---:|---:|
| `V3_User_Profile_L1_gpt56` | `User_Profile_L1_gpt56_V3` | Raw activities → interests | 287,557 | 4,985 |
| `V3_User_Profile_L1_Negative_gpt56` | `User_Profile_L1_Negative_gpt56_V3` | Negative feedback → rejected topics (V2 copy) | 557,734 | 5,000 |
| `V3_User_Profile_L2_gpt56` | `User_Profile_L2_gpt56_V3` | Existing profile + new interests → merge / add decisions | 259,606 | 4,977 |
| `V3_User_Profile_L3_Commercial_gpt56` | `User_Profile_L3_Commercial_gpt56_V3` | Interests → commercial label, brands, search queries | 201,304 | 4,998 |
| `V3_User_Profile_L3_Persona_gpt56` | `User_Profile_L3_Persona_gpt56_V3` | Interests → persona + category | 162,863 | 4,997 |
| `V3_User_Profile_L4_Biography_gpt56` | `User_Profile_L4_Biography_gpt56_V3` | Interests + facts → life stage, attributes, biography | 161,345 | 5,000 |
| `V3_User_Profile_L4_CommercialPreference_gpt56` | `User_Profile_L4_CommercialPreference_gpt56_V3` | Commercial interests → shopping preferences | 155,190 | 4,991 |
| `V3_User_Profile_L4_Hyper_gpt56` | `User_Profile_L4_Hyper_gpt56_V3` | Commercial interests + context → ad missions + new queries | 136,511 | 4,998 |

Train and test are split by user, so a user never appears in both splits of a config. L1 and L2 hold out 2% of
users (hash buckets 0–199 of 10,000); L3 and L4 hold out 4.85% (buckets 0–484), so no L1/L2 test user is in any
train split. Every test split is then capped at 5,000 rows: users are kept whole, in bucket order, until the next
user would exceed the cap, and the remaining held-out users are dropped (not moved to train). L1 Negative keeps its
V2 split by `uid` and is capped the same way.

| Config | Test rows / users before the cap | After the cap |
|---|---|---|
| L1 | 5,744 / 420 | 4,985 / 365 |
| L2 | 5,126 / 410 | 4,977 / 402 |
| L3 Commercial | 9,888 / 980 | 4,998 / 502 |
| L3 Persona | 8,051 / 978 | 4,997 / 606 |
| L4 Biography | 7,960 / 978 | 5,000 / 615 |
| L4 Commercial Preference | 7,739 / 978 | 4,991 / 633 |
| L4 Hyper | 6,813 / 1,287 | 4,998 / 942 |
| L1 Negative | 11,184 / 11,172 | 5,000 / 4,997 |

Full chat length (user + assistant) with the `Qwen/Qwen3.5-4B` chat template, thinking disabled, train and test
combined. Rows longer than 25,600 tokens are removed (L3 Commercial: 4,965 chunks; L4 Commercial Preference: 3,970
rows; no other config has such rows). Percentiles use nearest rank.

| Config | Users (train / test) | Mean | P50 | P90 | P99 | P99.9 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | 20,365 / 365 | 4,079 | 2,546 | 10,229 | 12,431 | 13,932 | 23,785 |
| L2 | 19,898 / 402 | 2,988 | 2,714 | 5,571 | 7,741 | 9,697 | 13,686 |
| L3 Commercial | 19,785 / 502 | 7,176 | 5,679 | 15,142 | 23,540 | 25,339 | 25,598 |
| L3 Persona | 19,779 / 606 | 3,361 | 3,108 | 6,342 | 8,653 | 10,567 | 15,627 |
| L4 Biography | 19,764 / 615 | 4,950 | 4,830 | 8,579 | 11,382 | 13,723 | 17,826 |
| L4 Commercial Preference | 19,730 / 633 | 7,214 | 5,562 | 16,197 | 23,777 | 25,393 | 25,600 |
| L4 Hyper | 25,911 / 942 † | 4,224 | 4,088 | 7,059 | 9,591 | 11,640 | 14,957 |
| L1 Negative | (V2 copy) | 442 | 352 | 551 | 2,063 | 3,900 | 10,444 |

† L4 Hyper calls carry no user ID. 93.5% of merged rows recover their user from the commercial preferences (see
L4 Hyper); the rest use a hash of the user context as `uid`, so their "users" are counted per context. The
L1 Negative token statistics are the V2 values (before the test cap).

## 2. What each row contains

All configs use the same outer row format:

```ts
{
  uid: string,             // 16-char hash: sha256(user_id)[:16]; L4 Hyper rows without a recovered user use a
                           // hash of the user context; L1 Negative hashes the row's actions (no user ID)
  messages: [
    { role: "user",      content: PROMPT + "\n\nInput:\n" + INPUT },
    { role: "assistant", content: OUTPUT }          // minimized JSON
  ],
  source: "GPT56",
  domain: string,          // see the table below
  think_type: "none",
  language?: string        // L1 and L1 Negative only, see the table below
}
```

Each row has one user message followed by one assistant message, with no system message. All rows in a config
use the same prompt, `prompts/prompt_*.md` verbatim. The assistant's response is a JSON string that can be read
with `json.loads`.

Use the following code to separate the prompt, input, and output:

```python
import json

prompt, _, raw_input = row["messages"][0]["content"].rpartition("\nInput:\n")
output = json.loads(row["messages"][1]["content"])
if row["domain"] in ("maiprofile-layer1-combined", "maiprofile-layer1-negative"):  # JSONL: one object per line
    input_data = [json.loads(line) for line in raw_input.split("\n")]
else:
    input_data = json.loads(raw_input)
```

| Config | `domain` | Prompt | `language` column | Input format |
|---|---|---|---|---|
| L1 | `maiprofile-layer1-combined` | `prompt_l1.md` | language of the input activities (metadata only, not in the output) | JSONL |
| L1 Negative | `maiprofile-layer1-negative` | `prompt_negative.md` | market: `en` or `glb` | JSONL |
| L2 | `maiprofile-layer2-combined` | `prompt_l2.md` | *(no column)* | JSON |
| L3 Commercial | `maiprofile-layer3-commercial` | `prompt_l3_commercial.md` | *(no column)* | JSON |
| L3 Persona | `maiprofile-layer3-persona` | `prompt_l3_persona.md` | *(no column)* | JSON |
| L4 Biography | `maiprofile-layer4-biography` | `prompt_l4_biography.md` | *(no column)* | JSON |
| L4 Commercial Preference | `maiprofile-layer4-commercial-preference` | `prompt_l4_commercial_preference.md` | *(no column)* | JSON |
| L4 Hyper | `maiprofile-layer4-hyper` | `prompt_l4_hyper.md` | *(no column)* | JSON |

The L1 `language` value is a fastText lid.176 vote over the row's activities (`mix` when no language covers 80% of
them; Chinese split into `zh-Hans` / `zh-Hant`). It is not shown to the model. V3 outputs are multilingual, so the
fixed `en` and the `query_language` values that V2 stored for L3/L4 no longer apply and those configs have no
`language` column.

## 3. How the tasks fit together

L1 extracts interests from activity history. L2 merges new interests into an existing profile. L3 adds persona
and commercial information, which the L4 tasks use to build a biography, shopping preferences, and ad missions.
L1 Negative is separate because its records do not have user IDs that can be joined to the other tasks.
No task predicts or reads an inferred intent in V3.

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

## 4. Reading the schemas

The schemas below use these shared types:

```ts
// Source: activity source
type Source = "MSN" | "Edge" | "Bing" | "Uet" | "ChromeImports" | "Ads" | "Shopping";
type Level  = "high" | "medium" | "low";
type Scale  = "Low" | "Medium-Low" | "Medium" | "Medium-High" | "High";
type Signal = "strong" | "moderate" | "weak";
```

V3 has no `Lang` type for outputs: interest names, activities, personas, entities, queries, and biographies are
written in the language of their own evidence (see each task).

In a schema, `field?` means the field is optional. A comment such as `~N%` tells you how often it is present.
Counts such as `0–39, median 7` describe the data we measured, not a limit imposed by the prompt.
All counts and percentages cover the full config, including train and test.

## L1: Interest extraction

Config: `V3_User_Profile_L1_gpt56`

The model reads a window of activities and extracts the user's interests. Each row merges the GPT-5.6 Final
`l1_delta` and `l1_actual` calls of one user-day window (there is no intent call).

**Input**

The input is JSONL, with one activity per line, sorted by date with `idx` renumbered from 0. A row contains
1–200, median 41 activities.

```ts
{ idx: int, date: "YYYY-MM-DD", source: Source, action: string }   // action: page title, query, or article title
```

**Output**

```ts
{
  interests: [{                             // 0–39, median 7 (0.98% of rows have none)
    topics: [{                              // 1–32, median 2
      evidence: int[],                      // input idx, sorted; 1–198, median 2
      topic: string                         // in the language of its activities
    }],
    interest_name: string,                  // in the majority language of the interest's activities
    actual_activity: string                 // factual description, in the language of interest_name
  }]
}
```

The keys are written in this order: evidence before topic, and topics before interest_name.
20.4% of interest names contain non-ASCII characters. Rows were dropped when
an interest's `actual_activity` is not written in the language of its `interest_name` (0.18% of rows).

The `language` column covers 49 values. The most common are:

| `language` | Train | Test |
|---|---:|---:|
| `en` | 168,392 | 3,017 |
| `mix` | 27,320 | 428 |
| `de` | 20,658 | 314 |
| `ja` | 16,801 | 255 |
| `fr` | 14,585 | 305 |
| `es` | 9,968 | 206 |
| `pt` | 9,856 | 169 |
| `zh-Hans` | 4,554 | 45 |
| `nl` | 4,549 | 82 |
| `zh-Hant` | 2,533 | 5 |
| `pl` | 1,912 | 27 |
| `it` | 1,572 | 48 |
| `sv` | 1,372 | 18 |
| others (36 values) | 3,485 | 66 |

## L1 Negative: Rejected topics

Config: `V3_User_Profile_L1_Negative_gpt56`

The train split is a byte-for-byte copy of `V2_User_Profile_L1_Negative_gpt56`; the test split keeps 5,000 of its
11,184 rows (section 1). The model reads a batch of negative feedback and identifies the topics the user wants to
avoid.

**Input**

The input is JSONL, with one feedback signal per line and no dates. A row contains 1–148 signals, with a median
of 1. Most rows (74%) contain just one signal.

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

Every input signal is cited by at least one negative interest. The input and output statistics are the V2 values.

## L2: Profile merge

Config: `V3_User_Profile_L2_gpt56`

For each new interest, the model decides whether to merge it into an existing interest or add it to the profile.
Names and activities follow the language of their topics, so a German delta can merge into an English snapshot
interest.

**Input**

```ts
{
  // existing profile, 0–90, median 22 interests
  snapshot: [{ interest_name: string, actual_activity: string, topics: string[] }],
  // new interests, 1–39, median 7
  delta:    [{ interest_name: string, actual_activity: string, topics: string[] }]
}
```

**Output**

Return one decision for each interest in `delta`. 99.99% of rows keep the input order.

```ts
{
  decisions: [
      { action: "merge",                     // 71.3%
        delta_interest_name: string, snapshot_interest_name: string,
        merged_interest_name: string, merged_actual_activity: string }
    | { action: "add",                       // 28.7%
        delta_interest_name: string, actual_activity: string }
  ]
}
```

Rows were dropped when a merged or added activity is not written in the language of its resulting name (0.24%).

## L3 Commercial: Commercial classification

Config: `V3_User_Profile_L3_Commercial_gpt56`

For each interest, the model decides whether it is relevant to advertisers. For every interest, commercial or not,
it also lists the brands, retailers, and products the evidence names and predicts search queries.

**Input**

A user with more than 30 interests is split into contiguous, near-equal chunks of at most 30 interests, one row per
chunk (the inference code calls the model the same way).

```ts
{
  interests: [{                              // 1–30, median 18
    interest_name: string,
    actual_activity: string,
    sources: Source[],
    topics: [{ topic: string, actions: string[] }]   // actions: raw activity texts (1–3)
  }]
}
```

**Output**

Return one entry for each input interest, in input order.

```ts
{
  interest_commercial: [{
    interest_name: string,
    commercial: boolean,                                       // true 75.4%
    commercial_score: "high" | "medium" | "low" | null,        // true: high 64.2%, medium 29.7%, low 6.1%
    intent_funnel_stage: "discovery" | "research" | "consideration" | "purchase" | "post-purchase" | null,
    brands: string[],
    retailers: string[],
    products: string[],
    predicted_queries: string[]                                // 0–3, in the language of the evidence actions
  }]
}
```

When `commercial` is `false`, `commercial_score` and `intent_funnel_stage` are `null`. Unlike V2, the entity and
query lists are not forced empty:

| | Commercial (2,703,875 entries) | Non-commercial (883,304 entries) |
|---|---:|---:|
| ≥ 1 brand | 75.1% | 21.6% |
| ≥ 1 retailer | 63.5% | 0.9% |
| ≥ 1 product | 83.4% | 1.1% |
| 3 / 2 / 1 / 0 queries | 81.1% / 18.1% / 0.9% / 0% | 61.7% / 31.9% / 6.4% / 2 entries |

Funnel stages of commercial entries: discovery 37.0%, consideration 31.3%, purchase 13.8%, research 10.2%, post-purchase 7.8%. Queries have
1–9 words (median 3);
19.0% contain non-ASCII characters. Queries keep the teacher's casing (official
brand spellings can be uppercase).

## L3 Persona: Interest personas

Config: `V3_User_Profile_L3_Persona_gpt56`

For each interest, the model describes the user's motivations and behavior and assigns a category path, both in
the language of the interest name.

**Input**

```ts
{
  interests: [{                 // 1–92, median 22
    interest_name: string,
    actual_activity: string,
    topics: string[]
  }]
}
```

**Output**

Return one entry for each input interest.

```ts
{
  interest_personas: [{
    interest_name: string,
    category: string,           // "/" path, broad → specific, 1–4 levels (median 2),
                                //   e.g. "/Casa e giardino/Ristrutturazione e manutenzione"
    persona: string             // 44.1% contain non-ASCII characters
  }]
}
```

Rows were dropped when a persona is not written in the language of its interest name (0.42%).

## L4 Biography: User attributes and biography

Config: `V3_User_Profile_L4_Biography_gpt56`

The model uses interests, facts, and declared attributes to infer the user's life stage and other attributes,
then writes a short biography.

This config has 166,345 rows from 20,379 users. The train split has 19,764 users and the test split has
615, with no users shared between the two splits.

As in V2, `user_attributes.age_group.signals` was removed from the GPT-5.6 output after validation (it is present
for every row without a declared `age_group`), and every attribute's `confidence_score` was removed.

**Input**

```ts
{
  declared_attributes: {                     // {} in 65.1% of rows
    age_group?: "18 - 24" | "25 - 34" | "35 - 49" | "50 - 65" | "65+",   // ~31.9%
    gender?: "Male" | "Female"                                           // ~25.9%
  },
  facts: {                                   // {} in 55.2% of rows
    market?: string,                         // ~28.3%, e.g. "en-US"
    location_data?: {                        // ~29.8%
      current_location: string,
      place_frequency:  [{ place: string,  count: int }],
      region_frequency?: [{ region: string, count: int }],   // ~97.4% of location_data
      recent_visits?: string[]                               // ~52.9% of location_data
    },
    linkedin_profile?: {                     // ~15.9%; sparse fields, each a string or string[]
      profile_geo?: string,
      industries?, degree?, degrees?, fieldOfStudy?, areas?, titles?, current_roles?,
      yearsOfExperience?, experience_band?, seniorities?, functions?, companies?, current_company?
    }
  },
  interests: [{                              // 1–92, median 22
    interest_name: string,
    actual_activity: string,
    persona: string,                         // from L3 Persona
    confidence_score: float,
    count: int,
    first_detect_date: "YYYY-MM-DD",
    last_detect_date: "YYYY-MM-DD",
    source: Source[],
    topics: string[],
    brands?: string[],                       // ~56.5%
    products?: string[],                     // ~62.7%
    predicted_queries?: string[]             // ~75.0%, from L3 Commercial
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
  biography: string                          // 52–566, median 314 chars
}
```

The stored JSON uses a fixed key order. Each `Attr` is written as `{evidence, value, confidence, source}`.
For `gender`, the order is `{lean, evidence, value, confidence, source}`. The evidence and biography follow the
user's language: 43.5% of biographies contain non-ASCII characters.

**Output statistics** (166,345 rows)

The following table shows the distribution of `life_stage.value`:

| `value` | Rows | Share |
|---|---:|---:|
| `unknown` | 93,935 | 56.5% |
| `retirement` | 25,439 | 15.3% |
| `parenting` | 21,221 | 12.8% |
| `job_seeking` | 16,483 | 9.9% |
| `single` | 5,505 | 3.3% |
| `caregiving` | 2,403 | 1.4% |
| `married` | 1,261 | 0.8% |
| `new_grad` | 98 | 0.1% |

Life-stage confidence is low in 79.0% of rows, medium in 17.1%, and high in 4.0%. The evidence list has 1 item in
25.3% of rows, 2 items in 59.9%, and 3 items in 14.8%.

The next table summarizes `user_attributes`. All percentages use the full row count:

| Attribute | `Unknown` | `source` (non-unknown) | `confidence` low / medium / high | Most common known values |
|---|---:|---|---|---|
| `age_group` | 7.2% | behavioral_inference 60.3%, declared 31.9%, linkedin 0.6% | 59.9 / 5.3 / 34.8% | `35 - 49` 49.8%, `50 - 65` 22.0%, `65+` 17.3%, `25 - 34` 2.5%, `18 - 24` 1.1%, `13 - 17` 57 rows |
| `gender` | 3.2% | behavioral_inference 70.9%, declared 25.9% | 37.9 / 36.2 / 25.9% | `Male` 58.2%, `Female` 38.5% |
| `estimated_income_level` | 3.3% | behavioral_inference 96.5%, linkedin 0.2% | 99.7 / 0.3 / 14 rows | `Medium` 95.8%, `High` 0.8%, `Low` 70 rows |
| `home_location` | 37.8% | location_data 29.8%, behavioral_inference 24.3%, linkedin 8.2% | 49.2 / 25.6 / 25.2% | 11,006 distinct; `United States` 1.2%, `United Kingdom` 0.8%, `France` 0.5% |
| `occupation` | 70.6% | behavioral_inference 20.8%, linkedin 8.6% | 90.3 / 0.7 / 8.9% | 8,066 distinct; `Business / Finance Professional` 3.5%, `Technology Professional` 1.3% |
| `company` | 96.1% | linkedin 3.3%, behavioral_inference 0.5% | 96.5 / 0.2 / 3.3% | 815 distinct; `Self-Employed` 0.4%, then ≤ 0.05% each |
| `career_stage` | 62.1% | behavioral_inference 29.2%, linkedin 8.7% | 87.7 / 3.4 / 8.9% | `Senior Professional` 20.2%, `Retired` 9.7%, `Entry Level` 2.7%, `Director/Executive` 1.6% |
| `education_area` | 76.6% | behavioral_inference 17.9%, linkedin 5.4% | 94.4 / 0.2 / 5.5% | 5,299 distinct; `Business / Accounting` 3.6%, `Health Sciences` 1.7% |

**How to read the attributes**

An unknown attribute always has `value: "Unknown"`, `source: "unknown"`, and `evidence: []`. These three values
occur together in every row. A known attribute has 1–3 evidence items.

Declared attributes are copied without changes. All 53,007 rows with a declared `age_group` return that value
with `source: "declared"`. The same is true for the 43,042 rows with a declared gender.

The `gender.lean` object always contains `male` and `female`, with 0–10 entries in each list. When gender is
inferred, entries have the form `"<interest name> (strong|moderate|weak)"`, such as `"Football (moderate)"`.
Across these entries, 48.6% are weak, 45.9% moderate, 4.8% strong, and 0.7% use another form. When gender is
declared, the lists are not scored: they contain one free-text entry in 42,073 rows, or are both empty in 969 rows.

Although `age_group.signals` was removed, the evidence text was left as it was. As a result, 15,876 rows (9.5%)
still mention "later-life" or "later life".

## L4 Commercial Preference: Shopping preferences

Config: `V3_User_Profile_L4_CommercialPreference_gpt56`

The model summarizes how the user shops, including sensitivity to price and discounts, subscription preferences,
shopping categories, and dietary restrictions. The input and output schemas are unchanged from V2.

**Input**

```ts
{
  life_stage: { value: string, confidence: Level, evidence: string[] },   // L4 Biography output
  commercial_interests: [{                   // 1–63, median 16
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
  discount_sensitivity:  { value: Scale | "Unknown", details: Detail<"sensitivity", Scale>[] },   // Unknown 40.3%
  price_sensitivity:     { value: Scale | "Unknown", details: Detail<"sensitivity", Scale>[] },   // Unknown 28.4%
  subscription_affinity: { value: Scale | "Unknown", details: Detail<"affinity", Scale>[] },      // Unknown 66.1%
  affinity: {
    shopping: {
      product_categories: [{ category: AdsVertical, description: string }],   // [] in 2.7% of rows
      shopper_type: { value: ShopperType[], details: Detail<"type", ShopperType>[] }
    },
    dining: {
      restrictions: { value: Diet[], details: Detail<"restriction", Diet>[] }   // value ["Unknown"] in 95.2%
    }
  }
}

type AdsVertical = "HomeGarden" | "TravelTourism" | "ComputersConsumerElectronics" | "RetailersGeneralMerchandise"
                 | "Finance" | "Apparel" | "Vehicles" | "InternetTelecom" | "SportsFitness" | "Health"
                 | "BeautyPersonalCare";     // Ads Vertical L1
type ShopperType = "ResearchDriven" | "DealDriven" | "BrandRetailerLoyal" | "NoveltySeeker" | "ValueDriven"
                 | "PremiumDriven" | "ConvenienceDriven"
                 | "ImpulseDriven";          // allowed, not observed in V3
type Diet        = "Unknown" | "Vegan" | "No Gluten" | "Vegetarian" | "Low-Carb" | "Keto" | "Dairy-Free" | "Halal"
                 | "Kosher" | "Pescatarian" | "No Nuts" | "Paleo"
                 | "No Shellfish";           // the last 5 are rare (235 occurrences; No Shellfish 2)
```

When a `value` is `"Unknown"`, the corresponding `details` list is empty (`[]`). The prompt sets no output language;
27.4% of rows have non-ASCII evidence text.

## L4 Hyper: Ad missions and queries

Config: `V3_User_Profile_L4_Hyper_gpt56`

The model groups commercial interests into goals, called missions, and writes new search queries for ad
retrieval. Each mission names the interests it came from.

Each row merges one GPT-5.6 Final mission-discovery call with the query-generation calls of its candidate missions.
Neither stage carries a user ID, so the calls are joined on the user context and the candidate mission name. The
pipeline often ran discovery several times on one input; the first run whose every candidate has a query-generation
answer is used, and each input yields at most one row. The user is recovered for 93.5% of merged rows by matching
the commercial preferences' `product_categories` to that user's L4 Commercial Preference answer.

**Input**

The input is a JSON object. All four fields are strings, even when they contain JSON or Markdown.

```ts
{
  personal_context: string,        // Markdown list or "": Market, Current location, Age group, Gender,
                                   //   Recent visits
  professional_context: string,    // Markdown list, "" in ~89.4%: Current company, Current roles, Functions,
                                   //   Industries, Seniorities, Years of experience, Degrees, Education areas
  commercial_preferences: string,  // a JSON string: parse it again (format below)
  commercial_interests: string     // Markdown, one block per interest (format below)
}
```

Parse the `commercial_preferences` string again to get the object below. It is a shorter version of the
L4 Commercial Preference output, with `confidence_score` in place of `details`.

```ts
{
  discount_sensitivity?:  { value: Scale, confidence_score: float },
  price_sensitivity?:     { value: Scale, confidence_score: float },
  subscription_affinity?: { value: Scale, confidence_score: float },
  affinity?: {                                   // missing in 4 rows
    shopping: {
      product_categories?: [{ category: string, description: string }],   // missing in 27 rows
      shopper_type?: { value: ShopperType[], confidence_score: float }
    },
    dining?: { restrictions: { value: Diet[], confidence_score: float } }
  }
}
```

The `commercial_interests` string contains one Markdown block per interest, using the format below. There is no
`- Intent:` line in V3.

```md
### Interest 1: <interest_name>
- Category: /A/B/C
- Persona: ...
- Activity: ...
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
  missions: [{                                   // 0–16, median 11 (355 rows have none)
    mission_name: string,                        // in the language of its source interests
    source_interests: string[],                  // exact input interest names
    scenarios: ("shopping" | "travel" | "hobbies" | "technology" | "learning" | "fitness" | "dining")[],
    predicted_brands: string[],                  // ≤ 3 new brands
    predicted_queries: string[]                  // observed: 1–4 queries, each 1–8 words (median 4)
  }]
}
```

As in V2, each query is a plain string: the teacher's per-query `value_type`, `source_query_refs`, `delta_source`,
`delta_evidence`, and `decision_change` fields and the per-mission `enrichment_sources` field are removed. No query
repeats an Existing query of the input: 4,273 such teacher queries were removed, together with the
679 missions they left without a query.

**Output statistics** (141,509 rows, 1,301,316 missions, 2,674,072 queries)

A row contains 0–16 missions, with a mean of 9.2 and a median of 11. Each mission has 1–4 queries, with a mean
of 2.05 and a median of 2.

A mission can have several scenarios. The share tagged with each scenario is: shopping 82.8%, travel 16.4%, hobbies 14.6%, technology 12.4%, learning 6.2%, fitness 3.6%, and dining 3.3%.

Most missions predict no new brands: 87.7% have 0, 8.6% have 1, 2.4% have 2, and 1.2% have 3.

32.6% of mission names and 23.8% of queries contain non-ASCII characters.

## 5. Changes from V2

| Area | V2 | V3 |
|---|---|---|
| Teacher data | GPT-5.6 ProcessData | GPT-5.6 Final `process_dataset` |
| Configs | 8 profile tasks + Ranker | 7 profile tasks + L1 Negative (V2 copy); no Ranker |
| Output language | mostly English; L3 Commercial / L4 Hyper queries in `query_language` | multilingual: each interest, persona, entity, query, and biography follows the language of its evidence |
| `language` column | L1 = predicted locale (output); L3 Commercial / L4 Hyper = `query_language`; other L3/L4 = `en` | L1 = detected input language (not in the output); L1 Negative = market; no column elsewhere |
| Inferred intent | L1 / L2 outputs; L3 Persona, L4 Biography, L4 Hyper inputs | removed everywhere |
| L1 output | `{predicted_content_locale, interests: [{interest_name, topics: [{topic, source, evidence}], actual_activity, inferred_intent}]}` | `{interests: [{topics: [{evidence, topic}], interest_name, actual_activity}]}` |
| L2 decisions | `merged_inferred_intent` / `inferred_intent` | removed |
| L3 Commercial input | `{interests, query_language}` | `{interests}`, chunks of ≤ 30 interests |
| L3 Commercial output | `commercial: false` → all lists `[]` | entities and queries for every interest |
| L3 Persona / L4 Biography input | interests carry `inferred_intent` | removed |
| L4 Hyper input | `query_language`; `- Intent:` lines | removed |
| L4 Hyper output | queries may repeat an Existing query of another interest | such queries removed |
| Prompts | numeric word / sentence / topic counts | no word, sentence, or topic count requirements |
| Split | 2% by user | L1/L2 2%, L3/L4 4.85% by user; every test split capped at 5,000 rows |
| Length filter | 20,480 (L1) / 25,600 tokens | 25,600 tokens |

## Appendix: Missing and empty input fields

This section covers the 7 V3 profile configs built from the Final data (L1 Negative is unchanged; see the V2
document). For each input field, we counted missing keys separately from keys with an empty value (`{}`, `[]`, or
`""`). For L4 Hyper, we also parsed the JSON strings and checked each line of the Markdown fields.

Every row can be parsed, contains one `user` message followed by one `assistant` message, and uses its config's
prompt. In the tables below, "rare" means fewer than 0.5% of the rows or items being counted.

### A.1 How each config represents missing data

| Config | Rows | Top-level keys (always all present) | Top-level value can be empty | Absent optional data inside |
|---|---:|---|---|---|
| L4 Hyper | 141,509 | `personal_context`, `professional_context`, `commercial_preferences`, `commercial_interests` | `personal_context` = `""` 54.8%; `professional_context` = `""` 89.4% | **line / key omitted** inside the Markdown and JSON strings (never an empty value) |
| L4 Commercial Preference | 160,181 | `life_stage`, `commercial_interests`, `non_commercial_interest_names` | `non_commercial_interest_names` = `[]` 15.4% | **key kept, value `[]`** (no key is ever missing) |
| L4 Biography | 166,345 | `declared_attributes`, `facts`, `interests` | `declared_attributes` = `{}` 65.1%; `facts` = `{}` 55.2% | **key omitted** (never `[]` / `""` / `{}` below the top level, except rare `persona` / `actual_activity` `""`) |
| L3 Commercial | 206,302 | `interests` | no | key kept; rare `""` |
| L3 Persona | 167,860 | `interests` | no | key kept; rare `""` |
| L2 | 264,583 | `snapshot`, `delta` | `snapshot` = `[]` in 18 rows | key kept; rare `""` |
| L1 | 292,542 | JSONL lines `{idx, date, source, action}` | no | every line has all 4 keys, none empty |

### A.2 L4 Hyper

**Context fields**

| `personal_context` | `professional_context` | Rows | Share |
|---|---|---:|---:|
| `""` | `""` | 77,542 | 54.80% |
| non-empty | `""` | 49,025 | 34.64% |
| non-empty | non-empty | 14,933 | 10.55% |
| `""` | non-empty | 9 | 0.01% |

**Commercial preferences**

The `commercial_preferences` value is always a non-empty JSON string. After parsing it, the top-level keys
appear in these combinations:

| Keys | Rows | Share |
|---|---:|---:|
| `affinity`, `discount_sensitivity`, `price_sensitivity` | 53,139 | 37.55% |
| `affinity` only | 29,684 | 20.98% |
| `affinity`, `discount_sensitivity`, `price_sensitivity`, `subscription_affinity` | 24,437 | 17.27% |
| `affinity`, `price_sensitivity` | 23,423 | 16.55% |
| `affinity`, `discount_sensitivity` | 5,642 | 3.99% |
| `affinity`, `price_sensitivity`, `subscription_affinity` | 3,795 | 2.68% |
| `affinity`, `subscription_affinity` | 895 | 0.63% |
| `affinity`, `discount_sensitivity`, `subscription_affinity` | 490 | 0.35% |
| `price_sensitivity` only / `subscription_affinity` only (no `affinity`) | 3 / 1 | 0.00% |

When `affinity` is present, it always contains `shopping`. Within it, `product_categories` is missing in
27 rows and `shopper_type` in 7.6%. The `dining` field appears in 1.7% of rows
(2,401 rows).

**Personal context**

The following labels appear in non-empty `personal_context` strings, across 63,958 rows:

| Label | Present | Notes |
|---|---:|---|
| `Gender` | 84.5% | `M` / `F` |
| `Current location` | 84.0% | free text, mixed formats |
| `Age group` | 77.2% | raw age or a letter code, not an age band |
| `Market` | 61.7% | e.g. `en-us` |
| `Recent visits` | 35.2% | comma-joined places |

The most common combinations are Age group, Current location, Gender, Market (21.6%); Age group, Current location, Gender, Market, Recent visits (20.9%); Current location (13.5%); Age group, Gender, Market (11.4%).
There are 22 combinations.

**Professional context**

There are 14,942 non-empty `professional_context` strings. The labels appear at these rates:
Industries 96.5%, Current roles 84.7%, Seniorities 84.2%, Functions 79.5%, Years of experience 73.0%, Degrees 53.2%, Education areas 53.1%, Current company 31.5%.

**Commercial interests**

There are 2,538,641 `commercial_interests` blocks. Each row has 1–69, median 17 blocks.

| Line | Present |
|---|---:|
| `### Interest N: <name>`, `Existing queries` | always |
| `Category`, `Persona`, `Activity` | ≥ 99.9% each (missing 1,869 / 1,869 / 84 blocks) |
| `Intent` | never (removed) |
| `Known entities` | 90.4% |
| ↳ `Brands` / `Products` / `Retailers` (within Known entities) | 83.6% / 92.7% / 70.9% |

### A.3 L4 Commercial Preference

Every key is present. When data is unavailable, its value is an empty array (`[]`) or an empty string (`""`):

| Path | Empty value |
|---|---|
| `non_commercial_interest_names` | `[]` in 24,688 rows (15.41%) |
| `commercial_interests[].brands` | `[]` 25.01% of interests |
| `commercial_interests[].products` | `[]` 16.68% |
| `commercial_interests[].retailers` | `[]` 36.74% |
| `commercial_interests[].persona` | `""` 1,344 (0.05%) |

`topics`, `topics[].evidence`, and `topics[].topic` are never empty. The `commercial_interests` list has
1–63 items. The `life_stage` object always contains `value`, `confidence`, and `evidence`; `value` is
`"unknown"` in 56.8% of rows.

### A.4 L4 Biography

Both `declared_attributes` and `facts` are always present. Each is `{}` when no data is available; otherwise,
it contains only the known fields. The table shows how often each combination occurs:

| `facts` \ `declared_attributes` | `{}` | `{age_group, gender}` | `{age_group}` | `{gender}` | Total |
|---|---:|---:|---:|---:|---:|
| `{}` | 90,398 | 707 | 521 | 132 | 91,758 |
| `{location_data, market}` | 4,142 | 14,040 | 6,240 | 1,472 | 25,894 |
| `{market}` | 1,601 | 5,775 | 2,660 | 1,388 | 11,424 |
| `{linkedin_profile}` | 9,944 | 734 | 235 | 114 | 11,027 |
| `{location_data}` | 1,418 | 5,657 | 2,926 | 825 | 10,826 |
| `{linkedin_profile, location_data, market}` | 438 | 4,936 | 1,247 | 493 | 7,114 |
| `{linkedin_profile, location_data}` | 222 | 4,335 | 790 | 349 | 5,696 |
| `{linkedin_profile, market}` | 118 | 1,801 | 403 | 284 | 2,606 |
| **Total** | 108,281 | 37,985 | 15,022 | 5,057 | 166,345 |

In 54.3% of rows, both objects are `{}`, so the model has only the interest history to work with.

**Location data**

When `facts.location_data` is present (49,530 rows), it always contains `current_location` and `place_frequency`.
The `region_frequency` field is missing in 2.6% of these objects, and `recent_visits` in 47.1%.

**LinkedIn data**

The `facts.linkedin_profile` object (26,443 rows) has 60 combinations of fields. In 36.6% of these objects, the only
field is `profile_geo`.

**Interest fields**

For non-commercial interests, `predicted_queries`, `products`, and `brands` are omitted rather than set to `[]`.
Across all 3,742,822 interests, 51.9% have all three fields, 25.0% have none, 10.8% have `predicted_queries` and
`products`, 7.7% have only `predicted_queries`, and 4.6% have `predicted_queries` and `brands`. Every interest has
`topics` and a non-empty `source`; `persona` is `""` in 2,527 interests and `actual_activity` in 164.

### A.5 L3, L2, L1

No input keys are missing in these configs. The table lists the uncommon empty values:

| Config | Empty values |
|---|---|
| L3 Commercial | `interests[].actual_activity` `""` 122; `sources`, `topics`, and `topics[].actions` never empty |
| L3 Persona | `interests[].actual_activity` `""` 164; `topics` and `topics[]` never empty |
| L2 | `snapshot` `[]` 18 rows; `delta[].actual_activity` `""` 1,998 (0.10%); `snapshot[].actual_activity` `""` 470; `topics` / `topics[]` never empty |
| L1 | none. 110 source mixes; the most common are `Uet` only 17.8% and `Ads+Bing+MSN+Uet` 10.3% |
