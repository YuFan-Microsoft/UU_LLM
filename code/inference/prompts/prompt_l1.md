Extract the user's interests from one window of activities.

An interest is a broad, reusable category such as American Football, Politics, or Home Improvement; a topic is a keyword, product, hobby, or entity from the activities, such as Seahawks under American Football. Group activities whose category and intent match, list each group's topics with the idx of every activity that supports them, and skip noisy or unsafe activities.

Write each interest's name and actual_activity in the majority language of that interest's own activities, and keep each topic in the original language of its activities. actual_activity is a factual description in present tense and third person ("Reads about ...", "Searches for ...") that covers all of the interest's activities with their exact names and describes what was observed, not motivation or intent.

Return minimized JSON, with evidence before topic and topics before interest_name:
`{"interests":[{"topics":[{"evidence":[0],"topic":"..."}],"interest_name":"...","actual_activity":"..."}]}`
Return `{"interests":[]}` when no valid interest remains.
