For each interest, decide whether advertisers could target its product or service category. Corporate or civic news, single-company stock news, sports coverage without tickets or merch, careers, government processes, religion, and pure curiosity are not commercial.

For a commercial interest, set `commercial_score` by buying signal strength, `intent_funnel_stage` by how far the user has gone, list `brands`, `retailers`, and `products` named in the evidence, and write up to 3 short lowercase `predicted_queries` in `query_language`, at least one with buying intent. For a non-commercial interest, use null and empty lists.

Return one entry per interest with its exact name, as minimized JSON:
`{"interest_commercial":[{"interest_name":"...","commercial":true,"commercial_score":"high","intent_funnel_stage":"research","brands":["..."],"retailers":["..."],"products":["..."],"predicted_queries":["..."]},{"interest_name":"...","commercial":false,"commercial_score":null,"intent_funnel_stage":null,"brands":[],"retailers":[],"products":[],"predicted_queries":[]}]}`
