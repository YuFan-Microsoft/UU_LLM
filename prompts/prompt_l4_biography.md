Infer the user's life stage and attributes, and write a short biography from their interests, facts, and declared attributes.

Life stage is single, married, parenting, caregiving, job_seeking, new_grad, retirement, or unknown, with a confidence and brief evidence. Copy declared attributes verbatim and infer the other user attributes, each with evidence, value, confidence, and source. The biography is a short factual summary. Write the evidence and biography in the main language of the user's interests. Do not speculate beyond the data.

Return minimized JSON:
`{"life_stage":{"value":"...","confidence":"high","evidence":["..."]},"user_attributes":{"<attribute>":{"evidence":["..."],"value":"...","confidence":"high","source":"..."}},"biography":"..."}`
