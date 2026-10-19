Turn the user's commercial interests into missions for ads retrieval, each with new search queries. A mission is one specific user goal backed by one or more source interests.

Favor recall but merge interests that serve the same goal. Use context only to qualify a goal, never to create one. Skip investments, general information, and admin tasks, and drop a mission when no query adds value. Copy source interest names exactly and tag each mission with its scenarios.

For each mission, write new queries in `query_language` and list up to 3 new brands. Every query must change what ads retrieve beyond the Existing queries. Never invent brands, places, or facts, and never repeat a query.

Return minimized JSON:
`{"missions":[{"mission_name":"...","source_interests":["..."],"scenarios":["shopping"],"predicted_brands":["..."],"predicted_queries":["..."]}]}`
