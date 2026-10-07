Extract the user's interests from one window of activities.

An interest is a stable category, activity, or entity with one clear need and recommendation pool. Use the most specific label that would still fit if the need returned in other wording. Single pages, models, and news angles are topics, not interests. Group activities only when domain, intent, and candidates match. Skip noise.

For each interest, list its `topics` with source names and supporting activity `idx`, write a factual one-sentence `actual_activity`, and an `inferred_intent` one step above it that stays close to the action when evidence is thin. Write in English. Set `predicted_content_locale` to the main language of the activities, or "mix".

Return minimized JSON:
`{"predicted_content_locale":"...","interests":[{"interest_name":"...","topics":[{"topic":"...","source":["..."],"evidence":[0]}],"actual_activity":"...","inferred_intent":"..."}]}`
