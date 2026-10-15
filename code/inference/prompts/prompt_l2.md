Merge the user's new (delta) interests into their existing profile (snapshot).

Merge a delta interest into the snapshot interest with the same stable domain, intent, and candidates, even if names differ. Otherwise add it as new. Merged fields summarize both.

Return one decision per delta interest, in order, with exact input names, as minimized JSON:
`{"decisions":[{"action":"merge","delta_interest_name":"...","snapshot_interest_name":"...","merged_interest_name":"...","merged_actual_activity":"...","merged_inferred_intent":"..."},{"action":"add","delta_interest_name":"...","actual_activity":"...","inferred_intent":"..."}]}`
