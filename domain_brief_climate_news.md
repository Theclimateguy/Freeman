Build a compact Freeman domain for climate-news monitoring from live RSS signals.

The agent will ingest a stream of climate, policy, emissions, extreme-weather, adaptation, and mitigation news.
The purpose of the domain is not sector-specific forecasting for one firm, but continuous world-state tracking of
climate-system stress, policy coordination, transition momentum, and macro-social disruption risk.

Model the domain as a compact causal world with:

- 3-5 actors such as governments, firms, households, international institutions, or the climate system
- 4-7 resources on interpretable 0-100 style scales
- 2-4 outcomes that capture broad regimes
- 3-8 causal edges with realistic signs

The world should be able to react sensibly to news about:

- emissions policy tightening or rollback
- international climate negotiations and cooperation failures
- extreme heat, drought, flood, wildfire, and sea-level impacts
- adaptation investment and resilience buildout
- energy transition acceleration or slowdown
- macro spillovers from climate shocks

Prefer outcomes like:

- coordinated_transition
- fragmented_adaptation
- climate_crisis
- policy_backlash

Prefer resource/state variables like:

- mitigation_capacity
- adaptation_capacity
- climate_hazard_pressure
- policy_coordination
- transition_investment
- social_stability

The desired semantics are:

- positive developments should raise the probability of coordinated transition
- worsening hazards and weak coordination should raise the probability of climate crisis or fragmented adaptation
- politically destabilizing transition stress can raise policy backlash

Keep the package compact, numerically stable, and suitable for repeated online updates from noisy news flow.
