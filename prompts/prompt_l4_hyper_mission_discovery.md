Group the user's commercial interests into missions for ads retrieval. A mission is one specific user goal backed by one or more source interests.

Favor recall but merge missions that serve the same goal. Use context only to qualify a goal, never to create one. Tag each mission with its scenarios. Skip investments, general information, and admin tasks.

Copy source interest names exactly, without numbering. Return at most 12 missions, or none, as minimized JSON:
`{"candidate_missions":[{"mission_name":"...","source_interests":["..."],"scenarios":["shopping"]}]}`
