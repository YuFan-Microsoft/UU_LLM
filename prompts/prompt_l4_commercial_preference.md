Summarize how the user shops from their commercial interests and life stage.

- `discount_sensitivity`: how much promotions drive decisions.
- `price_sensitivity`: how strongly cost constrains choices, not the spend tier.
- `subscription_affinity`: preference for recurring plans, which needs explicit subscription evidence.
- `product_categories`: Ads Vertical L1 categories the user clearly shops in, each naming preferred brands or retailers.
- `shopper_type`: archetypes the evidence clearly supports.
- `dining.restrictions`: dietary restrictions, or ["Unknown"] if none.

The first three are independent and use Low, Medium-Low, Medium, Medium-High, or High; mixed evidence means Medium, and no usable evidence means Unknown with no details. Back them, shopper_type, and dining restrictions with details that carry an area, a level, the evidence, and a signal_strength. Values must agree with their details.

Return minimized JSON:
`{"discount_sensitivity":{"value":"Medium-High","details":[{"area":"clothing","sensitivity":"Medium-High","evidence":"...","signal_strength":"strong"}]},"price_sensitivity":{"value":"Medium","details":[{"area":"home","sensitivity":"Medium","evidence":"...","signal_strength":"moderate"}]},"subscription_affinity":{"value":"Unknown","details":[]},"affinity":{"shopping":{"product_categories":[{"category":"ComputersConsumerElectronics","description":"..."}],"shopper_type":{"value":["ResearchDriven"],"details":[{"area":"electronics","type":"ResearchDriven","evidence":"...","signal_strength":"strong"}]}},"dining":{"restrictions":{"value":["Unknown"],"details":[]}}}}`
