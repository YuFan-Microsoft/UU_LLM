For each interest, decide whether advertisers could target its product or service category; for a commercial interest, also extract the brands, retailers, and products its evidence names and predict the user's next queries.

`commercial` is true when the interest maps to a purchasable product or service category. Corporate, financial, or civic news, single-company stock news, sports coverage without tickets, merch, or subscriptions, careers, government processes, religion, person-following, and pure curiosity are not commercial. For a commercial interest, set `commercial_score` by buying signal strength and `intent_funnel_stage` by how far the user has gone; otherwise both are null.

A non-commercial interest has all four lists empty. For a commercial interest, list only `brands`, `retailers`, and `products` the evidence clearly names (products as generic head nouns unless an official model name), and write 1 to 3 short `predicted_queries` grounded in the evidence, at least one with buying intent. Keep entities and queries in the language of their evidence actions and topics, without translation.

Return one entry per interest with its exact name, as minimized JSON:
`{"interest_commercial":[{"interest_name":"...","commercial":true,"commercial_score":"high","intent_funnel_stage":"research","brands":["..."],"retailers":["..."],"products":["..."],"predicted_queries":["..."]},{"interest_name":"...","commercial":false,"commercial_score":null,"intent_funnel_stage":null,"brands":[],"retailers":[],"products":[],"predicted_queries":[]}]}`
