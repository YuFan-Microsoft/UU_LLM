# MAI Profile V3 — Layer 2: Temporal Interest Classification

## Task
Determine the temporal pattern for each activity/interest based on the **nature of the interest**. The interest/activity could be either broad or specific, but the temporal classification should be based on what the interest *is* in the real world, not how frequently or recently the user engaged with it.

## Classification Guide

| Category          | Description and Examples                  |
|-------------------|------------------------------|
| Ephemeral         | Breaking news, viral media, one time action/behavior/activity, or one-off events. Triggered externally and fades quickly once stimulus disappears. Examples: Celebrity news event, viral meme, login, download a software, "air bnb near Seatac airport", "World Cup score last night" |
| ShortTerm         | Trends, seasonal topics, or temporary goals tied to transient lifestyle states. Examples: Election coverage, vacation planning, seasonal shopping, Hackathons, software downloads or installations, recipes for a specific dish, tooling under a broader topic, looking for specific products, deals or services |
| LongTerm          | Multi-phase personal projects or learning goals. Stable but still goal-driven. Examples: Ongoing geopolitical conflict, career development, professional development or skill learning, "machine learning tutorials" |
| Persistent        | Evergreen interests tied to identity, profession, enduring hobby or deep personal engagement. Never decays from inactivity alone. Examples: AI/technology, cooking, fitness, parenting, professional domain, skin care, gardening |

## Input Factors
- `interest_name`: The name of the interest/activity (e.g., "AI/technology", "vacation planning")
- `actual_activity`: The description of the activities related to the interest
- `topics`: The topics associated with the interest.
- `previous_name`: The previous interest name (if any), which can help identify if the interest has evolved or shifted focus.
- `previous_temporal`: The previous temporal classification for this interest (if any).

## Classification Guidelines
- **Classify based on the interest name's inherent nature, do not infer user's intent or engagement.** The temporal type reflects what the interest *is* in the real world, not how frequently or recently the user engaged with it.
- **Use domain knowledge and semantic meaning.** Ask: "What kind of thing is this interest?" A professional skill, a news event, a hobby, a seasonal activity? Or just an one-time activity? The answer determines the category. Download a software is an one-time activity, so it's Ephemeral. Cooking is a hobby, so it's Persistent. Election coverage is a seasonal topic, so it's ShortTerm. Being a fan of a specific sports team is a persistent interest, but following the live score of a specific game is ephemeral.
- **Use `topics` and `actual_activity` as disambiguation and evidence context** — but only to clarify *what the interest is*, not to measure engagement frequency. The topics only shows related contents included in the interest, not because of the activity frequency. The actual activity only shows the description of the interest.
- **Broader categories like "cooking", "classical music", "art" can be LongTerm or Persistent, while specific events like recipes for type of dish, concert for a specific artist, a specific activity or action or a song of an musician/artist are likely Ephemeral or ShortTerm.** The temporal type is determined by the inherent nature of the interest, not the user's engagement pattern.
- **Previous interest name can provide context for evolution or shift in focus.** - If `previous_name` and `previous_temporal` are provided, consider whether the interest has evolved or shifted focus, which may influence the temporal classification. For example, if an interest was previously "Flight to Seattle" (Ephemeral) but has evolved to "Seattle travel planning" (ShortTerm) as current `interest_name`, it should be reclassified as ShortTerm based on the broader and more enduring nature of the new interest name. Another example, if an interest was previously classified as ShortTerm due to a specific event (e.g., "2024 Olympics"), but the interest name has evolved to a broader category (e.g., "Olympic sports training"), it may warrant reclassification to LongTerm or Persistent based on the enduring nature of the new interest name. However, if the interest name remains the same and strongly indicates a specific temporal category (e.g., "2024 Olympics"), then the previous temporal classification should be retained unless there is compelling evidence to suggest otherwise.


## Output Format
Output is valid minimized JSON, no space, no newline.
```json
{
  "interests": [
    {
      "interest_name": "keep the same interest_name from input",
      "temporal": "Ephemeral|ShortTerm|LongTerm|Persistent",
      "reason": "one sentence explaining the reasoning behind the temporal classification, referencing the input factors and classification guide"
    }
  ]
}
```