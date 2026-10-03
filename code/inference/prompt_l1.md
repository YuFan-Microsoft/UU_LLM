Build one current-window user-interest profile from indexed activities.

The input groups activities by date under `days`; each row follows `columns`: [idx, source, action, intent], where `intent` is an upstream hint for that activity.

Choose stable, recommendation-ready interests, group concrete supporting topics, and omit unrelated noise. For each topic, copy its contributing source names and reference supporting activities with their integer `idx` values. Write one factual `actual_activity` sentence and one concise `inferred_intent` for each interest. Write all text in English, even when the activities are in another language. Set `predicted_content_locale` to the dominant language code of the activities (e.g. en, de, ja, zh-Hans), or "mix" if no single language dominates.

Return only minimized JSON:
`{"predicted_content_locale":"...","interests":[{"interest_name":"...","topics":[{"topic":"...","source":["..."],"evidence":[0]}],"actual_activity":"...","inferred_intent":"..."}]}`
