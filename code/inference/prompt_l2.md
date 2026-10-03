For every delta interest, decide whether to add it as a new profile interest or merge it into one existing snapshot interest with the same durable domain, intent, and recommendation candidate pool. Also classify the resulting interest as Ephemeral, ShortTerm, LongTerm, or Persistent based on the nature of the interest.

The input lists existing profile interests under `snapshot` and current interests under `delta`; each has `interest_name`, `actual_activity`, and `topics`.

Return exactly one decision per delta interest, in input order. Use exact input names for `delta_interest_name` and `snapshot_interest_name`. Merged fields must summarize both the snapshot and delta interest, and delta interests merged into the same snapshot interest must share one `merged_interest_name`.

Return only minimized JSON:
`{"decisions":[{"action":"merge","delta_interest_name":"...","snapshot_interest_name":"...","merged_interest_name":"...","merged_actual_activity":"...","merged_inferred_intent":"...","temporal":"LongTerm"},{"action":"add","delta_interest_name":"...","actual_activity":"...","inferred_intent":"...","temporal":"ShortTerm"}]}`
