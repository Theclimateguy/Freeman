<!-- Example brief only. This file is a domain-specific smoke-test asset for llm_synthesize bootstrap, not a required part of the generic runtime. -->

Build a compact Freeman domain for climate-news monitoring from live RSS, arXiv, and HTTP signal streams.

Assume the runtime already starts with a seeded ontology-heavy knowledge graph in `data/kg_climate_seed.json`.
Your job in `llm_synthesize` is not to reproduce that full ontology, but to synthesize a small, numerically stable
state-space model that compresses the live news flow into a few interpretable latent resources and outcomes.

The agent will ingest a stream of:

- emissions and decarbonization policy news
- disclosure and climate-finance news
- physical-hazard and disaster alerts
- climate science and attribution research
- adaptation and resilience investment updates

Model the domain as a compact causal world with:

- 3-5 actors such as governments, firms, households, international institutions, regulators, or the climate system
- 4-7 resources on interpretable 0-100 style scales
- 2-4 outcomes that capture broad regimes
- 3-8 causal edges with realistic signs

Use this canonical ontology as the compression target:

- `Greenhouse Gases -> Global Warming -> Sea Level Rise`
- `Global Warming -> Extreme Weather -> Physical Risk -> Economic Loss -> Financial Stability`
- `Mitigation Policy -> Emissions -> Global Warming`
- `Adaptation Investment -> Adaptation Gap -> Physical Risk`
- `Climate Disclosure -> Transition Plan Gap -> Transition Risk`
- `Carbon Pricing -> type_of -> Mitigation Policy`
- `TCFD Framework -> governs -> Climate Disclosure`

The world should react sensibly to news about:

- emissions policy tightening or rollback
- international climate negotiations and cooperation failures
- extreme heat, drought, flood, wildfire, cyclone, and sea-level impacts
- adaptation investment and resilience buildout
- energy transition acceleration or slowdown
- macro-financial spillovers from climate shocks
- disclosure, stress testing, and climate-finance regulation

Prefer outcomes like:

- coordinated_transition
- fragmented_adaptation
- climate_crisis
- policy_backlash

Prefer resource/state variables like:

- mitigation_capacity
- adaptation_capacity
- climate_hazard_pressure
- physical_risk_load
- transition_momentum
- policy_coordination
- financial_fragility

Desired semantics:

- positive mitigation, disclosure, and clean-tech developments should raise the probability of `coordinated_transition`
- worsening hazards, sea-level pressure, and adaptation shortfall should raise `climate_crisis`
- repeated local resilience investments with weak global coordination should raise `fragmented_adaptation`
- abrupt transition costs with distributional stress should raise `policy_backlash`

Signal interpretation heuristics:

- physical hazard alerts should mostly shock `climate_hazard_pressure`, `physical_risk_load`, or `financial_fragility`
- disclosure / supervisory news should mostly update `policy_coordination`, `transition_momentum`, or `financial_fragility`
- mitigation policy and clean-tech deployment should mostly raise `mitigation_capacity` and lower hazard pressure with lag
- adaptation and resilience finance should mostly raise `adaptation_capacity` and lower realized physical risk

Keep the package compact, numerically stable, and suitable for repeated online updates from noisy climate news flow.
