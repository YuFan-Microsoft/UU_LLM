For one candidate mission, write new search queries that change what ads should retrieve beyond the Existing queries.

Label each query explore, refine, or advance. Use context only when a supplied fact changes the results, and record what you evaluate in the audit blocks. Never invent brands, places, or facts. Drop the mission if nothing adds value.

Write 1-3 lowercase queries of 2-7 words (up to 4 when the mission has several source interests) in `query_language`, and up to 3 new brands. Every query cites the Existing queries it improves on and the source of its added value. Other fields stay in English.

Return minimized JSON:
`{"geo_resolution":{...},"professional_opportunities":[...],"price_tier_resolution":{...},"shopping_category_opportunities":[...],"preference_opportunities":[...],"enhanced_missions":[{"input_mission_name":"...","mission_name":"...","source_interests":["..."],"scenarios":["shopping"],"predicted_brands":["..."],"predicted_queries":[{"query":"...","value_type":"explore","source_query_refs":["..."],"delta_source":"commercial_preference","delta_evidence":"...","decision_change":"..."}],"enrichment_sources":["commercial_preference"]}]}`
