Extract the topics the user rejected from one batch of negative feedback signals.

Each signal's `action` is the feedback type, the advertiser for ads, and the ad or content title, separated by "|". A negative interest is one specific topic the user wants to avoid. Group signals only when they reject the same topic and share a detailed source. Cover every signal.

For each negative interest, write `keywords` as the rejected topic in at most 8 lowercase words, as a phrase or comma-separated keywords, never a reaction such as "not interested", "offensive", or "misleading". List the supporting signal `idx` as `evidence`, their unique `source` names, their `detailed_source`, their unique lowercase feedback types as `signal_type`, and a `reliability_score` from 0 to 1 for how clearly they reject the topic.

Return minimized JSON:
`{"negative_interests":[{"keywords":"...","evidence":[0],"source":["..."],"detailed_source":"...","signal_type":["downvote"],"reliability_score":0.9}]}`
