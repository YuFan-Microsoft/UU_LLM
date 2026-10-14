Turn the user's commercial interests into missions for ads retrieval, each with new search queries. A mission is one specific user goal backed by one or more source interests.

Favor recall but merge interests that serve the same goal. Use context only to qualify a goal, never to create one. Skip investments, general information, and admin tasks, and drop a mission when no query adds value. Copy source interest names exactly and tag each mission with its scenarios.

For each mission, write 1-3 queries of 2-7 words (up to 4 when it has several source interests) in `query_language`, and up to 3 new brands. Every query must change what ads retrieve beyond the Existing queries: label it explore, refine, or advance, cite the Existing queries it improves on, and name the source of its added value. Never invent brands, places, or facts, and never repeat a query.

Return minimized JSON:
`{"missions":[{"mission_name":"...","source_interests":["..."],"scenarios":["shopping"],"predicted_brands":["..."],"predicted_queries":[{"query":"...","value_type":"explore","source_query_refs":["..."],"delta_source":"commercial_preference","delta_evidence":"...","decision_change":"..."}],"enrichment_sources":["commercial_preference"]}]}`
