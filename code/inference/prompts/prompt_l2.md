Merge the user's new (delta) interests into their existing profile (snapshot).

Merge a delta interest into the one snapshot interest with the same category or user intent, even if the names or languages differ; otherwise add it as new. Use a reusable category that covers both sides, and keep distinct goals separate. A merge writes merged_interest_name and merged_actual_activity summarizing both sides without inventing actions, in the majority language of the merged topics; an add keeps the delta's activity.

Return exactly one decision per delta interest, with input names copied exactly, as minimized JSON:
`{"decisions":[{"action":"merge","delta_interest_name":"...","snapshot_interest_name":"...","merged_interest_name":"...","merged_actual_activity":"..."},{"action":"add","delta_interest_name":"...","actual_activity":"..."}]}`
