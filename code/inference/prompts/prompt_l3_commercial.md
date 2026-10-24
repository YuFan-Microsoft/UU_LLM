For each interest, decide whether advertisers could target its product or service category, extract the brands, retailers, and products its evidence names, and predict the user's next queries.

`commercial` is true when the interest maps to a purchasable product or service category. Corporate, financial, or civic news, single-company stock news, sports coverage without tickets, merch, or subscriptions, careers, government processes, religion, person-following, and pure curiosity are not commercial. For a commercial interest, set `commercial_score` by buying signal strength and `intent_funnel_stage` by how far the user has gone; otherwise both are null.

For every interest, commercial or not, list only `brands`, `retailers`, and `products` the evidence clearly names (products as generic head nouns unless an official model name), and write up to 3 short `predicted_queries` grounded in the evidence, at least one with buying intent for a commercial interest and informational otherwise. Keep entities and queries in the language of their evidence actions, without translation.

Return one entry per interest with its exact name, as minimized JSON:
`{"interest_commercial":[{"interest_name":"...","commercial":true,"commercial_score":"high","intent_funnel_stage":"research","brands":["..."],"retailers":["..."],"products":["..."],"predicted_queries":["..."]},{"interest_name":"...","commercial":false,"commercial_score":null,"intent_funnel_stage":null,"brands":[],"retailers":[],"products":[],"predicted_queries":["..."]}]}`
