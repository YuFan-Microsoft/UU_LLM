Summarize how the user shops from their commercial interests and life stage.

- `deal_seeking`: how much promotions drive decisions. Mixed evidence means medium.
- `price_tier`: the spend level the user leans toward. Premium or budget needs consistent signals.
- `product_categories`: Ads Vertical L1 categories the user clearly shops in, each naming preferred brands or retailers.
- `shopper_type`: archetypes the evidence clearly supports.
- `dining.restrictions`: dietary restrictions, or ["Unknown"] if none.

Back deal_seeking, price_tier, shopper_type, and dining restrictions with details that carry an area, a level, the evidence, and a signal_strength. Values must agree with their details.

Return minimized JSON:
`{"deal_seeking":{"value":"high","details":[{"area":"travel","seeking":"high","evidence":"...","signal_strength":"strong"}]},"price_tier":{"value":"premium","details":[{"area":"fashion","tier":"premium","evidence":"...","signal_strength":"moderate"}]},"affinity":{"shopping":{"product_categories":[{"category":"ComputersConsumerElectronics","description":"..."}],"shopper_type":{"value":["ResearchDriven"],"details":[{"area":"electronics","type":"ResearchDriven","evidence":"...","signal_strength":"strong"}]}},"dining":{"restrictions":{"value":["Unknown"],"details":[]}}}}`
