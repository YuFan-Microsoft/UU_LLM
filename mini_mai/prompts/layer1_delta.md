# MAI Profile V3 — Layer 1: Delta Interest Extraction

## Task

Extract recommendation-ready user interests from the interaction signals in this delta.

An **interest** is a stable category, activity, goal, or consumption domain that can be used directly for personalization. A **topic** is a concrete entity, product, person, destination, event, query theme, or finer facet that supports that interest.

**Core Principle:** One interest represents one durable user intent and one coherent recommendation/content candidate pool. It does not need to represent only one entity.

---

## Target Granularity

Choose the **canonical durable interest boundary that best explains the signals**: semantically stable across future deltas, broad enough to support a reusable range of future recommendations, content, or ads, and specific enough to imply one coherent user need and candidate pool. `Durable` describes the stability of the boundary, not how long the user's current engagement must last; a valid interest may represent a short-term need.

- Normalize every signal to the same semantic scale before creating interests. Do not let one signal become a page-level interest while another becomes a broad domain, and do not choose a finer name merely because it describes today's wording more literally.
- Prefer the most specific **reusable** abstraction that preserves the observed intent, supports a varied future candidate pool, and is likely to remain the same when the same need reappears through another page, phrasing, or item serving that same need. Stop broadening as soon as materially different user needs or candidate pools would enter the label.
- A single page, article, query, release, match result, price, or temporary news angle is normally a topic or evidence, not an interest.
- A broad umbrella that mixes substantially different user needs or candidate pools is too coarse.
- A recognizable sub-domain, activity, or product or service class is usually the right scale.
- A specific product model, feature, page, article, or isolated content object is normally a topic even when its name is stable. Stability alone is not enough: an interest must support multiple plausible future recommendations beyond more information about the same item.
- A durable product, platform, franchise, competition, destination, person, or other entity may itself be an interest only when it has an independent ongoing audience and a varied recommendation or content space, and moving to its parent would materially change the observed intent. Otherwise, use the stable parent domain and retain the entity as a topic.
- Generalize a concrete item only when the broader label preserves its apparent intent and does not imply unsupported interest in materially different siblings. Multiple sibling signals are helpful but not required when the broader intent is directly supported.
- Normally use 2–5 words. A precise one-word domain or durable entity is allowed.
- Put transient details and finer facets in `topics`. Do not invent sibling topics or unsupported broader intent merely to make an interest coarser.

First decide whether a signal supports a qualified interest boundary at all. Admit it when the observed activity can be represented as a stable category, activity, goal, or qualified durable entity without inventing a broader intent. Omit it when removing the page, item, fact, or temporary task leaves no supported reusable boundary. Signal count is not an admission rule: one clear signal can support a stable category such as a product or service class, while many weak signals may still support no useful interest. A concrete entity does not qualify merely because its name is stable, well known, or associated with an independent audience; the observed activity must itself show a reusable consumption, participation, or engagement need centered on that entity. A one-off fact, schedule, status, or location lookup about an entity is normally a topic or may be omitted unless the evidence supports a broader reusable boundary.

### Boundary tests

For each proposed interest, apply all three tests:

1. **Too-narrow test:** If the page, query wording, date, version, feature, article angle, result, or temporary event changed, would this still be the same user-interest boundary? Could the name support several meaningfully different future recommendations instead of only more information about the same item? If either answer is no, move the detail to `topics` and use the nearest stable parent that preserves intent.
2. **Cross-delta stability test:** If the same underlying need appeared in a later delta through different wording, another page, or another item serving the same need, would it naturally map to this same boundary and canonical name? If not, the current boundary is probably tied too closely to today's surface form. Move that detail to a topic and choose the nearest supported reusable boundary. This test does not authorize inventing interest in unsupported siblings, and a qualified durable entity may remain its own boundary.
3. **Too-broad test:** Would the proposed name naturally include substantially different user goals or recommendation candidates that the evidence does not support? If yes, keep the more specific domain, activity, or durable entity.

The correct interest is the most specific reusable boundary that passes all three tests and has a coherent future candidate pool. The number of signals or topics does not determine granularity: one clear signal can support a durable interest when its meaning supports that boundary, while many signals must still remain separate when their intents or candidate pools differ.

The examples below illustrate scale, not required labels:

| Good interest | Supporting topics | Why this scale works |
|---|---|---|
| `AI Assistants` | Microsoft Copilot, ChatGPT, Claude | One assistant-use candidate pool |
| `Developer Tools` | GitHub Copilot, OpenAI Codex, Visual Studio Code | One development workflow |
| `Cloud Computing` | Microsoft Azure, AWS, Cloud Compute | Stable technical domain |
| `FIFA World Cup` | 2026 World Cup, Match Tickets, National Teams | Durable competition audience |
| `Air Travel` | Flight Booking, Airline Fares, Airport Transfers | Coherent travel need |
| `Business Travel` | Seattle Hotel, Corporate Shuttle, Flight Booking | Coherent goal and context |

Avoid both extremes:

| Too narrow | Better | Too broad |
|---|---|---|
| `ChatGPT Pricing Page` | `ChatGPT` for product-specific intent, or `AI Assistants` for cross-product intent | `Technology` |
| `GitHub Copilot Release Notes` | `Developer Tools` when the evidence shows a development workflow | `Technology` |
| `2026 World Cup Final Score` | `FIFA World Cup` | `Sports` |
| `Azure Outage Article` | `Azure` for platform-specific intent, or `Cloud Computing` for cross-platform intent | `Technology` |
| `Safe Driver Discount Page` | `Vehicle Insurance` | `Finance` |
| `Hilton Seattle Room Page` | `Business Travel` when the evidence shows a business-trip goal | `Travel` |
| `Blocked Kitchen Sink` | `Home Plumbing Repair` | `Home & Garden` |
| `Luxury Perfume Collection Page` | `Luxury Fragrance` | `Beauty & Personal Care` |

The target is a semantic band, not a fixed hierarchy or vocabulary. The examples illustrate boundary behavior and are not required labels. A single clear signal may establish a stable category or qualified durable entity; omit a weak, ambiguous, navigational, or isolated fact/task only when no reusable boundary is supported without speculation.

---

## Grouping Rules

Review the full delta before emitting any interest. First map all signals to candidate durable interests, then consolidate candidates globally. Do not finalize interests one signal at a time.

Group signals into the same interest only when all three conditions hold:

1. **Domain fit:** the topics belong to one specific stable sub-category, activity, or durable entity audience. Sharing only a broad ancestor is not enough.
2. **Intent fit:** the user's apparent goal or mode of engagement is compatible.
3. **Candidate-pool fit:** substantially the same recommendations or content could serve the topics.

Entity identity can establish that signals concern the same subject, but it does not by itself establish the correct interest granularity. The entity must still pass the independent-audience and varied-future-candidate tests. Shared brand, company, retailer, source, location, or a loose keyword is not sufficient.

Assign each signal/topic bundle to one **primary interest owner** before finalizing the output:

- Do not copy the same topic or evidence into parent, child, sibling, or differently worded interests merely to express multiple levels of a hierarchy.
- The same entity may appear under different interests only when the supporting evidence, including different signals or clearly separable aspects of a complex signal, shows materially different user intents or candidate pools. In that case, qualify each topic name by its role or intent rather than repeating the same ambiguous topic name.
- When one set of signals expresses one need, choose one durable boundary as its owner and retain the narrower entities, products, pages, and facets as topics under it.

After grouping, compare every pair of proposed interests:

- If one is only a feature, event, query angle, or temporary facet of the other, keep one interest and move the narrower material into its topics.
- If one candidate can only recommend more information about the same model, feature, page, or isolated object, absorb it into the nearest stable parent that preserves intent.
- If two names differ but describe the same enduring user need and candidate pool, consolidate them under one canonical name.
- If a parent and child would both use the same evidence, keep only the most specific durable boundary that covers that evidence.
- Keep both only when each has a materially distinct intent or candidate pool; wording differences alone are not a reason to split.

### Keep separate

- `Microsoft Stock` and `Microsoft Copilot`: same company, different intents and candidate pools.
- `Developer Tools` and `Network Security`: different recommendation domains.
- `Japan Travel` and `Peru Travel`: separate concrete trips unless the signals clearly express one broader recurring international-travel behavior.
- `Home Buying` and `Home Improvement`: related life domain, different user goals.
- `Taylor Swift` and unrelated music artists: shared industry alone is too broad.

### Entity and event handling

- Prefer putting concrete entities and transient details in `topics`.
- A durable entity may be `interest_name` only when it has an independent ongoing audience and varied recommendation or content space, and its parent would materially change the observed intent.
- Broaden an entity to a parent category only when the evidence supports that broader intent and substantially the same candidate pool. Retain the concrete entity as a topic.
- Fold news, releases, prices, rumors, match results, and updates into the enduring interest they reinforce. Use those temporary angles in the topic name only when necessary for fidelity.
- Do not create both a parent interest and its child interest from the same signals in the same delta.

---

## Naming Rules

### Interest names

- Name the stable domain or user activity, not the observed page or temporary event.
- Use natural, self-explanatory English.
- Avoid generic suffixes such as `News`, `Updates`, `Research`, `Content`, or `Interest` unless that is genuinely the repeated activity.
- Avoid catch-all conjunctions such as `Business and Technology News`.

### Topic names

- Use concise, self-explanatory names, usually 2–5 words.
- Preserve the concrete entity or subject needed to understand the topic.
- Topics under an interest may be different entities when they pass the domain, intent, and candidate-pool tests.

### Source names

- Use the source names from the input signals, including but not limited to `MSN`, `Bing`, `Edge`, `Uet`, `Ads`, `Copilot`, `Xbox`, and `Linkedin`.
- List every contributing source once.

---

## Process

1. Translate non-English signals to English.
2. Remove weak navigational, functional, ambiguous, or one-off noise that does not support a useful interest.
3. Review all signals together and identify concrete entities/topics and their apparent user intents.
4. Map each signal to a canonical durable boundary using all three boundary tests.
5. Group signals by that boundary, intent, and candidate pool.
6. Assign every signal/topic bundle one primary interest owner, then consolidate overlapping candidate groups globally.
7. Choose one natural canonical name for each surviving group.
8. Extract 1–5 self-explanatory topics and attach the evidence owned by that group.
9. Compare all output interests again and remove parent/child, facet/domain, differently worded, and duplicate-topic ownership conflicts.

---

## Output Format

Output valid minimized JSON with no surrounding explanation.

```json
{
  "interests": [
    {
      "interest_name": "Stable Interest Domain",
      "topics": [
        {
          "topic": "Concrete Supporting Topic",
          "source": ["Edge", "Bing"],
          "evidence": [
            {
              "date": "YYYY-MM-DD",
              "source": ["Edge"],
              "detailed_source": "Edge ChromeExt",
              "action": "raw signal text",
              "intent": "upstream per-signal intent hint"
            }
          ]
        }
      ]
    }
  ]
}
```

If no valid interests exist, return `{"interests":[]}`.

---

## Final Checklist

- [ ] Each interest is broader than an individual signal but specific enough to imply one coherent user need and candidate pool.
- [ ] Each interest supports a reusable, varied future recommendation/content/ad space beyond more information about one item.
- [ ] Every interest passes the too-narrow, cross-delta stability, and too-broad tests.
- [ ] Each interest is at least as durable as its individual topics.
- [ ] Grouped topics share domain, user intent, and recommendation candidate pool.
- [ ] Transient details are topics; an entity remains an interest only when it has an independent ongoing audience, varied future candidates, and a parent that would materially change intent.
- [ ] No unsupported or mixed-intent umbrella buckets, mixed-intent groups, or parent/child duplicates.
- [ ] Every signal/topic bundle has one primary owner; repeated entities across interests have distinct evidence, intent, and intent-qualified topic names.
- [ ] The full delta was consolidated globally; no same-domain fragment survives merely because it came from a different signal.
- [ ] Evidence volume did not determine granularity; weak isolated facts/tasks were omitted, while clear single-signal categories were retained.
- [ ] Output is valid minimized JSON.
