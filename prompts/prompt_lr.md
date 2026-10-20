Rank the user's interests by how useful their predicted queries are likely to be for retrieving the user's future activities.

Use actual_activity, inferred_intent, topic names, and detection dates to judge which interests are likely to continue. Consider recency and clear intent, and whether the queries match the activity summaries. Do not invent behavior or treat missing information as evidence.

Build the ranking one interest at a time. At each step, choose the interest whose queries are likely to retrieve the most activities not already covered by earlier picks. Discount overlap with interests already selected. The candidates are shuffled; input order and idx values do not indicate priority.

Select exactly 10 interests, ordered from the greatest expected new coverage to the least. Copy their input idx values exactly, with no repeats. Return only output_idx, without names, scores, or explanations.

Return minimized JSON:
`{"output_idx":[4,1,7,0,9,3,6,2,8,5]}`
