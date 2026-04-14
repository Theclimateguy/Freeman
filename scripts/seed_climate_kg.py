"""Build a deterministic climate seed knowledge graph for Freeman.

The seed graph is intentionally ontology-heavy: it front-loads domain concepts,
causal chains, risk channels, policy levers, and measurement nodes so that the
runtime can start from a dense graph instead of constructing one only from live
signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

SEED_TIMESTAMP = "2026-04-14T00:00:00+00:00"
SEED_VERSION = "climate_seed_v1"
DEFAULT_SEED_PATH = Path("data/kg_climate_seed.json")
DEFAULT_MEMORY_PATH = Path("data/kg_climate.json")

SOURCE_REFERENCES = {
    "ipcc_ar6": {
        "label": "IPCC AR6 Synthesis Report",
        "url": "https://www.ipcc.ch/report/ar6/syr/",
    },
    "owid_climate": {
        "label": "Our World in Data: CO2 and Greenhouse Gas Emissions",
        "url": "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
    },
    "wikipedia_climate": {
        "label": "Wikipedia Climate Change Portal",
        "url": "https://en.wikipedia.org/wiki/Portal:Climate_change",
    },
}

BUCKET_SOURCES = {
    "root": ["ipcc_ar6", "owid_climate", "wikipedia_climate"],
    "forcing": ["ipcc_ar6", "owid_climate"],
    "response": ["ipcc_ar6", "owid_climate"],
    "hazard": ["ipcc_ar6", "wikipedia_climate"],
    "exposure": ["ipcc_ar6", "wikipedia_climate"],
    "sector": ["ipcc_ar6", "wikipedia_climate"],
    "impact": ["ipcc_ar6", "owid_climate"],
    "finance": ["ipcc_ar6", "owid_climate"],
    "policy": ["ipcc_ar6", "wikipedia_climate"],
    "adaptation": ["ipcc_ar6", "wikipedia_climate"],
    "technology": ["ipcc_ar6", "owid_climate"],
    "framework": ["ipcc_ar6", "wikipedia_climate"],
    "metric": ["owid_climate", "ipcc_ar6"],
    "actor": ["ipcc_ar6", "wikipedia_climate"],
    "scenario": ["ipcc_ar6", "owid_climate"],
}


def _source_payload(keys: list[str]) -> list[str]:
    return [f"{SOURCE_REFERENCES[key]['label']} ({SOURCE_REFERENCES[key]['url']})" for key in keys]


CORE_NODES: list[dict[str, Any]] = [
    {
        "id": "climate:climate_domain",
        "label": "Climate Domain",
        "node_type": "domain",
        "content": "Root ontology node for Freeman's climate seed graph.",
        "bucket": "root",
    },
    {
        "id": "climate:forcing_driver",
        "label": "Forcing Driver",
        "node_type": "concept",
        "content": "Drivers that alter atmospheric composition, land cover, or radiative balance.",
        "bucket": "forcing",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:greenhouse_gases",
        "label": "Greenhouse Gases",
        "node_type": "concept",
        "content": "Heat-trapping gases that increase effective radiative forcing.",
        "bucket": "forcing",
        "parent": "climate:forcing_driver",
    },
    {
        "id": "climate:forcing_process",
        "label": "Forcing Process",
        "node_type": "concept",
        "content": "Mechanisms linking emissions and land-use change to Earth-system imbalance.",
        "bucket": "forcing",
        "parent": "climate:forcing_driver",
    },
    {
        "id": "climate:earth_system_response",
        "label": "Earth System Response",
        "node_type": "concept",
        "content": "Persistent physical responses of the climate system to forcing.",
        "bucket": "response",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:global_warming",
        "label": "Global Warming",
        "node_type": "state",
        "content": "Long-run increase in global mean surface temperature from anthropogenic forcing.",
        "bucket": "response",
        "parent": "climate:earth_system_response",
    },
    {
        "id": "climate:sea_level_rise",
        "label": "Sea Level Rise",
        "node_type": "state",
        "content": "Increase in mean sea level through thermal expansion and land-ice loss.",
        "bucket": "response",
        "parent": "climate:earth_system_response",
    },
    {
        "id": "climate:physical_hazard",
        "label": "Physical Hazard",
        "node_type": "concept",
        "content": "Acute or chronic climate hazards arising from Earth-system change.",
        "bucket": "hazard",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:extreme_weather",
        "label": "Extreme Weather",
        "node_type": "concept",
        "content": "Extreme weather and climate events intensified by warming and hydrological change.",
        "bucket": "hazard",
        "parent": "climate:physical_hazard",
    },
    {
        "id": "climate:exposure_channel",
        "label": "Exposure Channel",
        "node_type": "concept",
        "content": "Transmission channels through which hazards reach assets, people, and systems.",
        "bucket": "exposure",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:sector",
        "label": "Sector",
        "node_type": "concept",
        "content": "Economic or social sector used to organize exposure and transition pathways.",
        "bucket": "sector",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:socioeconomic_impact",
        "label": "Socioeconomic Impact",
        "node_type": "concept",
        "content": "Real-economy consequences of hazard realization and adaptation gaps.",
        "bucket": "impact",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:physical_risk",
        "label": "Physical Risk",
        "node_type": "risk",
        "content": "Expected losses from climate hazards acting on exposed and vulnerable systems.",
        "bucket": "impact",
        "parent": "climate:socioeconomic_impact",
    },
    {
        "id": "climate:economic_loss",
        "label": "Economic Loss",
        "node_type": "impact",
        "content": "Output, wealth, or fiscal losses triggered by physical or transition shocks.",
        "bucket": "impact",
        "parent": "climate:socioeconomic_impact",
    },
    {
        "id": "climate:financial_risk",
        "label": "Financial Risk",
        "node_type": "concept",
        "content": "Financial-system vulnerabilities linked to physical and transition channels.",
        "bucket": "finance",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:transition_risk",
        "label": "Transition Risk",
        "node_type": "risk",
        "content": "Losses arising from policy, technology, litigation, and preference shifts in decarbonization.",
        "bucket": "finance",
        "parent": "climate:financial_risk",
    },
    {
        "id": "climate:financial_stability",
        "label": "Financial Stability",
        "node_type": "state",
        "content": "System-wide resilience of banks, insurers, markets, and public balance sheets.",
        "bucket": "finance",
        "parent": "climate:financial_risk",
    },
    {
        "id": "climate:policy_instrument",
        "label": "Policy Instrument",
        "node_type": "concept",
        "content": "Mitigation, adaptation, disclosure, and supervisory tools used in climate governance.",
        "bucket": "policy",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:mitigation_policy",
        "label": "Mitigation Policy",
        "node_type": "policy",
        "content": "Policies that reduce emissions or accelerate decarbonization.",
        "bucket": "policy",
        "parent": "climate:policy_instrument",
    },
    {
        "id": "climate:adaptation_measure",
        "label": "Adaptation Measure",
        "node_type": "policy",
        "content": "Actions that reduce vulnerability, exposure, or realized physical losses.",
        "bucket": "adaptation",
        "parent": "climate:policy_instrument",
    },
    {
        "id": "climate:climate_disclosure",
        "label": "Climate Disclosure",
        "node_type": "policy",
        "content": "Standards and reporting practices that surface climate exposures and transition plans.",
        "bucket": "framework",
        "parent": "climate:policy_instrument",
    },
    {
        "id": "climate:climate_finance",
        "label": "Climate Finance",
        "node_type": "policy",
        "content": "Capital allocation toward mitigation, adaptation, and resilience investments.",
        "bucket": "policy",
        "parent": "climate:policy_instrument",
    },
    {
        "id": "climate:transition_technology",
        "label": "Transition Technology",
        "node_type": "concept",
        "content": "Technologies that lower carbon intensity or increase system flexibility.",
        "bucket": "technology",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:disclosure_framework",
        "label": "Disclosure Framework",
        "node_type": "concept",
        "content": "Reporting or scenario-analysis frameworks used in climate risk governance.",
        "bucket": "framework",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:metric",
        "label": "Metric",
        "node_type": "concept",
        "content": "Observed indicators used to track forcing, impacts, or risk.",
        "bucket": "metric",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:actor",
        "label": "Actor",
        "node_type": "concept",
        "content": "Institutions, firms, and social groups that emit, regulate, or bear climate risk.",
        "bucket": "actor",
        "parent": "climate:climate_domain",
    },
    {
        "id": "climate:scenario",
        "label": "Scenario",
        "node_type": "concept",
        "content": "Macro climate-risk regimes used by Freeman for compact world-state interpretation.",
        "bucket": "scenario",
        "parent": "climate:climate_domain",
    },
]

CHILDREN_BY_PARENT: dict[str, list[dict[str, str]]] = {
    "climate:greenhouse_gases": [
        {"id": "climate:carbon_dioxide", "label": "Carbon Dioxide (CO2)", "content": "Long-lived greenhouse gas dominated by fossil-fuel combustion and land-use change.", "bucket": "forcing"},
        {"id": "climate:methane", "label": "Methane (CH4)", "content": "Potent short-lived greenhouse gas linked to energy, agriculture, and waste.", "bucket": "forcing"},
        {"id": "climate:nitrous_oxide", "label": "Nitrous Oxide (N2O)", "content": "Greenhouse gas strongly tied to fertilizer use and agricultural systems.", "bucket": "forcing"},
        {"id": "climate:fluorinated_gases", "label": "Fluorinated Gases", "content": "Synthetic greenhouse gases with very high global warming potential.", "bucket": "forcing"},
        {"id": "climate:black_carbon", "label": "Black Carbon", "content": "Short-lived climate forcer that warms the atmosphere and darkens snow and ice.", "bucket": "forcing"},
        {"id": "climate:tropospheric_ozone", "label": "Tropospheric Ozone", "content": "Short-lived greenhouse gas and air pollutant formed from precursor emissions.", "bucket": "forcing"},
        {"id": "climate:land_use_change", "label": "Land-Use Change", "content": "Deforestation and land conversion that alter carbon stocks and surface properties.", "bucket": "forcing"},
        {"id": "climate:cement_process_emissions", "label": "Cement Process Emissions", "content": "Industrial process emissions from clinker production and cement demand.", "bucket": "forcing"},
    ],
    "climate:forcing_process": [
        {"id": "climate:anthropogenic_emissions", "label": "Anthropogenic Emissions", "content": "Aggregate human-caused greenhouse-gas emissions flow.", "bucket": "forcing"},
        {"id": "climate:radiative_forcing", "label": "Radiative Forcing", "content": "Change in Earth's energy balance induced by atmospheric composition and surface change.", "bucket": "forcing"},
        {"id": "climate:earth_energy_imbalance", "label": "Earth Energy Imbalance", "content": "Excess retained energy in the Earth system after forcing changes.", "bucket": "response"},
        {"id": "climate:carbon_cycle_feedback", "label": "Carbon Cycle Feedback", "content": "Feedbacks that weaken natural carbon uptake or release additional carbon.", "bucket": "response"},
        {"id": "climate:aerosol_masking", "label": "Aerosol Masking", "content": "Short-lived cooling effect from aerosols that partially offsets warming.", "bucket": "forcing"},
        {"id": "climate:albedo_change", "label": "Albedo Change", "content": "Surface reflectivity change from snow, ice, land cover, or soot deposition.", "bucket": "response"},
        {"id": "climate:ocean_acidification", "label": "Ocean Acidification", "content": "Decline in ocean pH as seawater absorbs atmospheric carbon dioxide.", "bucket": "response"},
        {"id": "climate:net_zero_gap", "label": "Net-Zero Gap", "content": "Difference between pledged decarbonization and the trajectory consistent with net zero.", "bucket": "policy"},
    ],
    "climate:earth_system_response": [
        {"id": "climate:ocean_heat_content", "label": "Ocean Heat Content", "content": "Accumulated heat stored in the ocean from persistent energy imbalance.", "bucket": "response"},
        {"id": "climate:cryosphere_loss", "label": "Cryosphere Loss", "content": "Retreat of glaciers, snow cover, and polar ice under warming.", "bucket": "response"},
        {"id": "climate:glacier_mass_loss", "label": "Glacier Mass Loss", "content": "Net glacier ice loss that contributes to runoff changes and sea level rise.", "bucket": "response"},
        {"id": "climate:ice_sheet_instability", "label": "Ice Sheet Instability", "content": "Dynamic ice-sheet losses that increase long-run sea-level risk.", "bucket": "response"},
        {"id": "climate:arctic_sea_ice_decline", "label": "Arctic Sea Ice Decline", "content": "Loss of Arctic sea ice extent and thickness.", "bucket": "response"},
        {"id": "climate:permafrost_thaw", "label": "Permafrost Thaw", "content": "Warming-driven thaw that damages infrastructure and can release greenhouse gases.", "bucket": "response"},
        {"id": "climate:hydrological_intensification", "label": "Hydrological Intensification", "content": "Acceleration of the water cycle with stronger extremes in wet and dry conditions.", "bucket": "response"},
        {"id": "climate:atmospheric_moisture", "label": "Atmospheric Moisture", "content": "Higher moisture-holding capacity in a warmer atmosphere.", "bucket": "response"},
        {"id": "climate:mean_temperature_anomaly", "label": "Mean Temperature Anomaly", "content": "Deviation of global mean temperature from a baseline period.", "bucket": "metric"},
        {"id": "climate:marine_heat_content_anomaly", "label": "Marine Heat Content Anomaly", "content": "Ocean heat anomaly used to track heat uptake and marine stress.", "bucket": "metric"},
        {"id": "climate:ecosystem_regime_shift", "label": "Ecosystem Regime Shift", "content": "Persistent ecological restructuring induced by climate stress.", "bucket": "response"},
        {"id": "climate:climate_sensitivity", "label": "Climate Sensitivity", "content": "Long-run temperature response to a doubling of atmospheric CO2.", "bucket": "response"},
    ],
    "climate:extreme_weather": [
        {"id": "climate:heatwave", "label": "Heatwave", "content": "Extreme hot period with elevated mortality, labor, and infrastructure impacts.", "bucket": "hazard"},
        {"id": "climate:marine_heatwave", "label": "Marine Heatwave", "content": "Persistent extreme ocean heat that stresses marine ecosystems.", "bucket": "hazard"},
        {"id": "climate:drought", "label": "Drought", "content": "Extended moisture deficit affecting water, crops, and ecosystems.", "bucket": "hazard"},
        {"id": "climate:flash_drought", "label": "Flash Drought", "content": "Rapid-onset drought caused by heat and evapotranspiration spikes.", "bucket": "hazard"},
        {"id": "climate:wildfire_weather", "label": "Wildfire Weather", "content": "Meteorological conditions conducive to wildfire ignition and spread.", "bucket": "hazard"},
        {"id": "climate:heavy_precipitation", "label": "Heavy Precipitation", "content": "High-intensity rainfall events associated with flood risk.", "bucket": "hazard"},
        {"id": "climate:river_flood", "label": "River Flood", "content": "Flooding from rivers exceeding channel capacity.", "bucket": "hazard"},
        {"id": "climate:pluvial_flood", "label": "Pluvial Flood", "content": "Surface flooding from rainfall overwhelming drainage systems.", "bucket": "hazard"},
        {"id": "climate:coastal_flood", "label": "Coastal Flood", "content": "Flooding of coastal zones from sea-level rise, tides, and storms.", "bucket": "hazard"},
        {"id": "climate:storm_surge", "label": "Storm Surge", "content": "Abnormal sea-level rise generated by storms and cyclones.", "bucket": "hazard"},
        {"id": "climate:tropical_cyclone_intensity", "label": "Tropical Cyclone Intensity", "content": "Wind intensity and destructive potential of tropical cyclones.", "bucket": "hazard"},
        {"id": "climate:tropical_cyclone_rainfall", "label": "Tropical Cyclone Rainfall", "content": "Extreme cyclone-related precipitation and inland flood potential.", "bucket": "hazard"},
        {"id": "climate:compound_event", "label": "Compound Event", "content": "Simultaneous or sequential hazards that amplify total damage.", "bucket": "hazard"},
        {"id": "climate:crop_heat_stress", "label": "Crop Heat Stress", "content": "Direct yield damage from extreme heat during sensitive growth stages.", "bucket": "hazard"},
        {"id": "climate:water_scarcity", "label": "Water Scarcity", "content": "Chronic mismatch between water demand and reliable supply.", "bucket": "hazard"},
        {"id": "climate:vector_borne_disease_risk", "label": "Vector-Borne Disease Risk", "content": "Expansion of climate-sensitive disease vectors into new regions.", "bucket": "hazard"},
        {"id": "climate:coral_bleaching", "label": "Coral Bleaching", "content": "Thermal stress event that damages coral reef ecosystems.", "bucket": "hazard"},
        {"id": "climate:landslide_risk", "label": "Landslide Risk", "content": "Slope failure risk intensified by heavy rainfall and land degradation.", "bucket": "hazard"},
    ],
    "climate:exposure_channel": [
        {"id": "climate:coastal_asset_exposure", "label": "Coastal Asset Exposure", "content": "Exposure of buildings and infrastructure located in coastal zones.", "bucket": "exposure"},
        {"id": "climate:urban_heat_burden", "label": "Urban Heat Burden", "content": "Heat exposure amplified by urban form and limited cooling access.", "bucket": "exposure"},
        {"id": "climate:water_stress", "label": "Water Stress", "content": "Pressure on water systems from scarcity, drought, or demand growth.", "bucket": "exposure"},
        {"id": "climate:food_system_exposure", "label": "Food System Exposure", "content": "Exposure of food production, logistics, and prices to climate shocks.", "bucket": "exposure"},
        {"id": "climate:infrastructure_exposure", "label": "Infrastructure Exposure", "content": "Physical exposure of roads, grids, ports, and telecom systems.", "bucket": "exposure"},
        {"id": "climate:health_system_exposure", "label": "Health System Exposure", "content": "Population and care-system exposure to heat, smoke, and disease stress.", "bucket": "exposure"},
        {"id": "climate:labor_exposure", "label": "Labor Exposure", "content": "Worker exposure to heat, smoke, and outdoor hazard conditions.", "bucket": "exposure"},
        {"id": "climate:housing_exposure", "label": "Housing Exposure", "content": "Exposure of residential stock to flood, heat, and wildfire risk.", "bucket": "exposure"},
        {"id": "climate:supply_chain_exposure", "label": "Supply Chain Exposure", "content": "Dependence of production networks on climate-vulnerable nodes and routes.", "bucket": "exposure"},
        {"id": "climate:ecosystem_service_exposure", "label": "Ecosystem Service Exposure", "content": "Exposure of pollination, water filtration, and natural protection services.", "bucket": "exposure"},
        {"id": "climate:tourism_exposure", "label": "Tourism Exposure", "content": "Exposure of travel demand and destination viability to climate shifts.", "bucket": "exposure"},
        {"id": "climate:insurance_penetration_gap", "label": "Insurance Penetration Gap", "content": "Shortfall between insurable losses and actual insurance coverage.", "bucket": "finance"},
        {"id": "climate:grid_exposure", "label": "Grid Exposure", "content": "Exposure of power systems to heat, storms, fires, and hydrological variability.", "bucket": "exposure"},
        {"id": "climate:agricultural_exposure", "label": "Agricultural Exposure", "content": "Exposure of cropland, livestock, and farm incomes to climate shocks.", "bucket": "exposure"},
        {"id": "climate:forest_exposure", "label": "Forest Exposure", "content": "Exposure of forests to drought, pests, wildfire, and heat stress.", "bucket": "exposure"},
        {"id": "climate:port_exposure", "label": "Port Exposure", "content": "Exposure of maritime gateways to flood, surge, and heat disruption.", "bucket": "exposure"},
    ],
    "climate:sector": [
        {"id": "climate:agriculture", "label": "Agriculture", "content": "Sector exposed to heat, drought, flood, pests, and transition policy.", "bucket": "sector"},
        {"id": "climate:energy", "label": "Energy", "content": "Sector spanning fossil production, power systems, and decarbonization technologies.", "bucket": "sector"},
        {"id": "climate:transport", "label": "Transport", "content": "Sector exposed to fuel transitions, weather disruption, and infrastructure constraints.", "bucket": "sector"},
        {"id": "climate:real_estate", "label": "Real Estate", "content": "Sector exposed to flood, heat, insurance, and disclosure repricing.", "bucket": "sector"},
        {"id": "climate:insurance_sector", "label": "Insurance Sector", "content": "Sector underwriting catastrophe, health, and liability exposures.", "bucket": "sector"},
        {"id": "climate:banking_sector", "label": "Banking Sector", "content": "Sector transmitting climate shocks through credit, collateral, and liquidity channels.", "bucket": "sector"},
        {"id": "climate:public_finance_sector", "label": "Public Finance", "content": "Fiscal sector exposed to disaster relief, adaptation needs, and tax-base erosion.", "bucket": "sector"},
        {"id": "climate:water_utilities", "label": "Water Utilities", "content": "Sector exposed to drought, contamination, and capital-intensive adaptation.", "bucket": "sector"},
        {"id": "climate:healthcare", "label": "Healthcare", "content": "Sector stressed by heat, smoke, disease, and emergency response demand.", "bucket": "sector"},
        {"id": "climate:manufacturing", "label": "Manufacturing", "content": "Sector exposed to supply chains, energy costs, and carbon-intensity constraints.", "bucket": "sector"},
        {"id": "climate:forestry", "label": "Forestry", "content": "Sector exposed to wildfire, pests, drought, and carbon-market policy.", "bucket": "sector"},
        {"id": "climate:coastal_infrastructure", "label": "Coastal Infrastructure", "content": "Ports, seawalls, roads, and utilities exposed to surge and sea-level rise.", "bucket": "sector"},
    ],
    "climate:physical_risk": [
        {"id": "climate:chronic_heat_risk", "label": "Chronic Heat Risk", "content": "Persistent risk from rising mean temperatures and repeated heatwaves.", "bucket": "impact"},
        {"id": "climate:acute_flood_risk", "label": "Acute Flood Risk", "content": "Short-run destructive risk from river, pluvial, or coastal flood events.", "bucket": "impact"},
        {"id": "climate:wildfire_risk", "label": "Wildfire Risk", "content": "Expected losses from wildfire ignition, spread, smoke, and suppression costs.", "bucket": "impact"},
        {"id": "climate:water_supply_risk", "label": "Water Supply Risk", "content": "Risk of inadequate water delivery for households, industry, and ecosystems.", "bucket": "impact"},
        {"id": "climate:crop_yield_risk", "label": "Crop Yield Risk", "content": "Risk of climate-driven losses in agricultural yields and farm income.", "bucket": "impact"},
        {"id": "climate:health_risk", "label": "Health Risk", "content": "Risk of excess morbidity and mortality from heat, smoke, and disease.", "bucket": "impact"},
        {"id": "climate:infrastructure_damage_risk", "label": "Infrastructure Damage Risk", "content": "Risk of physical damage to infrastructure capital stocks.", "bucket": "impact"},
        {"id": "climate:ecosystem_loss_risk", "label": "Ecosystem Loss Risk", "content": "Risk of biodiversity loss and degradation of ecosystem services.", "bucket": "impact"},
        {"id": "climate:coastal_inundation_risk", "label": "Coastal Inundation Risk", "content": "Risk of permanent or episodic coastal flooding and erosion.", "bucket": "impact"},
        {"id": "climate:power_system_risk", "label": "Power System Risk", "content": "Risk of outages, derating, and system failures due to climate stress.", "bucket": "impact"},
        {"id": "climate:migration_risk", "label": "Migration Risk", "content": "Risk of displacement and migration driven by unmanageable local impacts.", "bucket": "impact"},
        {"id": "climate:supply_chain_risk", "label": "Supply Chain Risk", "content": "Risk of cascading disruption in production and logistics networks.", "bucket": "impact"},
        {"id": "climate:food_security_risk", "label": "Food Security Risk", "content": "Risk that climate shocks impair access, availability, or affordability of food.", "bucket": "impact"},
        {"id": "climate:water_quality_risk", "label": "Water Quality Risk", "content": "Risk that heat, flood, and runoff degrade potable and ecological water quality.", "bucket": "impact"},
    ],
    "climate:economic_loss": [
        {"id": "climate:gdp_loss", "label": "GDP Loss", "content": "Aggregate output loss from climate shocks and transition friction.", "bucket": "impact"},
        {"id": "climate:asset_impairment", "label": "Asset Impairment", "content": "Write-down of productive or residential assets after climate damage.", "bucket": "impact"},
        {"id": "climate:productivity_loss", "label": "Productivity Loss", "content": "Reduction in labor or capital productivity from heat and disruption.", "bucket": "impact"},
        {"id": "climate:fiscal_pressure", "label": "Fiscal Pressure", "content": "Higher public spending needs and weaker tax revenues from climate shocks.", "bucket": "impact"},
        {"id": "climate:inflation_pressure", "label": "Inflation Pressure", "content": "Inflation impulse from food, energy, insurance, and logistics shocks.", "bucket": "impact"},
        {"id": "climate:trade_disruption", "label": "Trade Disruption", "content": "Losses from interrupted trade corridors and higher transport costs.", "bucket": "impact"},
        {"id": "climate:commodity_price_spike", "label": "Commodity Price Spike", "content": "Large increase in food, power, or resource prices after climate shocks.", "bucket": "impact"},
        {"id": "climate:insured_loss", "label": "Insured Loss", "content": "Losses borne by insurance balance sheets after covered events.", "bucket": "impact"},
        {"id": "climate:uninsured_loss", "label": "Uninsured Loss", "content": "Losses borne directly by households, firms, or governments.", "bucket": "impact"},
        {"id": "climate:adaptation_gap", "label": "Adaptation Gap", "content": "Difference between required resilience and realized adaptation effort.", "bucket": "adaptation"},
    ],
    "climate:transition_risk": [
        {"id": "climate:policy_shock_risk", "label": "Policy Shock Risk", "content": "Risk from abrupt tightening of climate policy and regulatory standards.", "bucket": "finance"},
        {"id": "climate:technology_disruption_risk", "label": "Technology Disruption Risk", "content": "Risk of incumbent value destruction from cleaner technologies.", "bucket": "finance"},
        {"id": "climate:reputation_risk", "label": "Reputation Risk", "content": "Risk from stakeholder pressure over climate alignment and disclosure quality.", "bucket": "finance"},
        {"id": "climate:litigation_risk", "label": "Litigation Risk", "content": "Risk from climate-related lawsuits, liability claims, and legal precedent.", "bucket": "finance"},
        {"id": "climate:stranded_asset_risk", "label": "Stranded Asset Risk", "content": "Risk that high-carbon assets lose economic value before the end of life.", "bucket": "finance"},
        {"id": "climate:carbon_cost_risk", "label": "Carbon Cost Risk", "content": "Risk from direct and indirect carbon pricing or compliance costs.", "bucket": "finance"},
        {"id": "climate:demand_shift_risk", "label": "Demand Shift Risk", "content": "Risk from changing consumer demand toward lower-carbon products.", "bucket": "finance"},
        {"id": "climate:supply_side_transition_risk", "label": "Supply-Side Transition Risk", "content": "Risk from production bottlenecks in decarbonization supply chains.", "bucket": "finance"},
        {"id": "climate:disclosure_gap_risk", "label": "Disclosure Gap Risk", "content": "Risk from inadequate climate reporting and scenario transparency.", "bucket": "finance"},
        {"id": "climate:transition_plan_gap", "label": "Transition Plan Gap", "content": "Shortfall between announced and credible decarbonization plans.", "bucket": "finance"},
    ],
    "climate:financial_stability": [
        {"id": "climate:insurance_losses", "label": "Insurance Losses", "content": "Underwriting and claims losses from realized catastrophe events.", "bucket": "finance"},
        {"id": "climate:mortgage_default_risk", "label": "Mortgage Default Risk", "content": "Higher mortgage delinquency risk after property damage or repricing.", "bucket": "finance"},
        {"id": "climate:credit_risk", "label": "Credit Risk", "content": "Elevated borrower default probability from climate shocks.", "bucket": "finance"},
        {"id": "climate:market_repricing", "label": "Market Repricing", "content": "Rapid asset repricing triggered by new climate information or policy.", "bucket": "finance"},
        {"id": "climate:liquidity_stress", "label": "Liquidity Stress", "content": "Funding strain when climate losses or collateral calls spike.", "bucket": "finance"},
        {"id": "climate:sovereign_spread_widening", "label": "Sovereign Spread Widening", "content": "Higher sovereign financing costs due to fiscal and disaster vulnerability.", "bucket": "finance"},
        {"id": "climate:municipal_bond_stress", "label": "Municipal Bond Stress", "content": "Local-government funding strain from repeated hazard losses.", "bucket": "finance"},
        {"id": "climate:reserve_adequacy_pressure", "label": "Reserve Adequacy Pressure", "content": "Pressure on insurer and bank reserves under rising tail losses.", "bucket": "finance"},
        {"id": "climate:collateral_repricing", "label": "Collateral Repricing", "content": "Decline in collateral value after hazard discovery or transition shock.", "bucket": "finance"},
        {"id": "climate:solvency_pressure", "label": "Solvency Pressure", "content": "Capital adequacy stress from accumulating climate-related losses.", "bucket": "finance"},
        {"id": "climate:macro_financial_feedback", "label": "Macro-Financial Feedback", "content": "Amplification loop between losses, credit contraction, and weaker activity.", "bucket": "finance"},
        {"id": "climate:fiscal_financial_linkage", "label": "Fiscal-Financial Linkage", "content": "Transmission of public-balance-sheet stress into financial-system risk.", "bucket": "finance"},
    ],
    "climate:mitigation_policy": [
        {"id": "climate:carbon_pricing", "label": "Carbon Pricing", "content": "Tax or cap-and-trade instrument that prices carbon externalities.", "bucket": "policy"},
        {"id": "climate:emissions_standard", "label": "Emissions Standard", "content": "Regulatory limit on emissions intensity or total emissions.", "bucket": "policy"},
        {"id": "climate:methane_regulation", "label": "Methane Regulation", "content": "Policy targeting methane leaks, venting, and agricultural emissions.", "bucket": "policy"},
        {"id": "climate:coal_phaseout", "label": "Coal Phaseout", "content": "Policy schedule to retire unabated coal generation or mining.", "bucket": "policy"},
        {"id": "climate:renewable_subsidy", "label": "Renewable Subsidy", "content": "Fiscal support for renewable generation deployment.", "bucket": "policy"},
        {"id": "climate:clean_power_standard", "label": "Clean Power Standard", "content": "Mandate requiring a rising share of low-carbon electricity.", "bucket": "policy"},
        {"id": "climate:ev_mandate", "label": "EV Mandate", "content": "Vehicle regulation that accelerates electric vehicle adoption.", "bucket": "policy"},
        {"id": "climate:building_efficiency_code", "label": "Building Efficiency Code", "content": "Code that lowers building energy demand and operating emissions.", "bucket": "policy"},
        {"id": "climate:deforestation_control", "label": "Deforestation Control", "content": "Policy to slow land-use emissions and preserve carbon sinks.", "bucket": "policy"},
        {"id": "climate:industrial_decarbonization_policy", "label": "Industrial Decarbonization Policy", "content": "Support and standards for low-carbon industry and process heat.", "bucket": "policy"},
        {"id": "climate:ndc_ambition", "label": "NDC Ambition", "content": "Strength of national climate commitments under the Paris Agreement.", "bucket": "policy"},
        {"id": "climate:loss_and_damage_fund", "label": "Loss and Damage Fund", "content": "International financing mechanism for unavoidable climate damages.", "bucket": "policy"},
        {"id": "climate:climate_stress_testing", "label": "Climate Stress Testing", "content": "Supervisory scenario exercises for climate-related financial risk.", "bucket": "framework"},
        {"id": "climate:green_taxonomy", "label": "Green Taxonomy", "content": "Classification system defining environmentally sustainable activities.", "bucket": "policy"},
    ],
    "climate:adaptation_measure": [
        {"id": "climate:adaptation_investment", "label": "Adaptation Investment", "content": "Capital expenditure dedicated to resilience and vulnerability reduction.", "bucket": "adaptation"},
        {"id": "climate:early_warning_system", "label": "Early Warning System", "content": "Monitoring and alerting system that reduces disaster mortality and losses.", "bucket": "adaptation"},
        {"id": "climate:flood_defense", "label": "Flood Defense", "content": "Levees, barriers, and drainage upgrades reducing flood damages.", "bucket": "adaptation"},
        {"id": "climate:drought_resilient_crops", "label": "Drought-Resilient Crops", "content": "Crop varieties with lower yield sensitivity to water stress.", "bucket": "adaptation"},
        {"id": "climate:wildfire_fuel_management", "label": "Wildfire Fuel Management", "content": "Land and vegetation management that reduces wildfire spread.", "bucket": "adaptation"},
        {"id": "climate:urban_cooling", "label": "Urban Cooling", "content": "Cooling centers, tree cover, and design that reduce urban heat risk.", "bucket": "adaptation"},
        {"id": "climate:water_reuse", "label": "Water Reuse", "content": "Reuse of treated water to strengthen supply reliability.", "bucket": "adaptation"},
        {"id": "climate:desalination", "label": "Desalination", "content": "Technology that augments water supply in scarcity-prone regions.", "bucket": "adaptation"},
        {"id": "climate:resilient_grid", "label": "Resilient Grid", "content": "Grid hardening and flexibility that lower outage risk.", "bucket": "adaptation"},
        {"id": "climate:managed_retreat", "label": "Managed Retreat", "content": "Planned relocation away from unmanageable coastal or flood risk.", "bucket": "adaptation"},
        {"id": "climate:resilient_building_codes", "label": "Resilient Building Codes", "content": "Codes that reduce losses from wind, flood, heat, and wildfire.", "bucket": "adaptation"},
        {"id": "climate:catastrophe_insurance", "label": "Catastrophe Insurance", "content": "Risk transfer instrument that smooths disaster recovery financing.", "bucket": "adaptation"},
        {"id": "climate:nature_based_solutions", "label": "Nature-Based Solutions", "content": "Ecosystem-based resilience measures such as wetland and mangrove restoration.", "bucket": "adaptation"},
        {"id": "climate:heat_health_action_plan", "label": "Heat-Health Action Plan", "content": "Operational plan for heat advisories, outreach, and response.", "bucket": "adaptation"},
        {"id": "climate:irrigation_efficiency", "label": "Irrigation Efficiency", "content": "Water-saving irrigation that reduces drought exposure.", "bucket": "adaptation"},
        {"id": "climate:climate_services", "label": "Climate Services", "content": "Decision-support services based on forecasts, risk maps, and advisories.", "bucket": "adaptation"},
    ],
    "climate:transition_technology": [
        {"id": "climate:solar_pv", "label": "Solar PV", "content": "Zero-operational-emissions power generation technology.", "bucket": "technology"},
        {"id": "climate:wind_power", "label": "Wind Power", "content": "Low-carbon electricity generation from onshore and offshore wind.", "bucket": "technology"},
        {"id": "climate:battery_storage", "label": "Battery Storage", "content": "Flexibility technology that supports renewable integration.", "bucket": "technology"},
        {"id": "climate:grid_transmission", "label": "Grid Transmission", "content": "Transmission buildout needed for power-system decarbonization.", "bucket": "technology"},
        {"id": "climate:heat_pumps", "label": "Heat Pumps", "content": "Electrified heating and cooling technology with high efficiency.", "bucket": "technology"},
        {"id": "climate:green_hydrogen", "label": "Green Hydrogen", "content": "Hydrogen produced with low-carbon electricity for hard-to-abate sectors.", "bucket": "technology"},
        {"id": "climate:nuclear_power", "label": "Nuclear Power", "content": "Low-carbon firm generation technology.", "bucket": "technology"},
        {"id": "climate:carbon_capture_storage", "label": "Carbon Capture and Storage", "content": "Technology that captures and stores CO2 from point sources.", "bucket": "technology"},
        {"id": "climate:direct_air_capture", "label": "Direct Air Capture", "content": "Technology that removes carbon dioxide directly from ambient air.", "bucket": "technology"},
        {"id": "climate:demand_response", "label": "Demand Response", "content": "Load-flexibility mechanism that lowers peak demand and supports clean grids.", "bucket": "technology"},
        {"id": "climate:building_retrofits", "label": "Building Retrofits", "content": "Capital upgrades that lower building energy demand and vulnerability.", "bucket": "technology"},
        {"id": "climate:low_carbon_cement", "label": "Low-Carbon Cement", "content": "Process and materials innovation that reduces cement-sector emissions.", "bucket": "technology"},
        {"id": "climate:low_carbon_steel", "label": "Low-Carbon Steel", "content": "Steelmaking pathways with materially lower carbon intensity.", "bucket": "technology"},
        {"id": "climate:sustainable_aviation_fuel", "label": "Sustainable Aviation Fuel", "content": "Lower-carbon fuel option for aviation transition.", "bucket": "technology"},
    ],
    "climate:climate_disclosure": [
        {"id": "climate:transition_plan", "label": "Transition Plan", "content": "Time-bound plan that links targets, capex, and governance to decarbonization.", "bucket": "framework"},
        {"id": "climate:scope_1_emissions", "label": "Scope 1 Emissions", "content": "Direct greenhouse-gas emissions from owned or controlled sources.", "bucket": "metric"},
        {"id": "climate:scope_2_emissions", "label": "Scope 2 Emissions", "content": "Indirect emissions from purchased electricity and heat.", "bucket": "metric"},
        {"id": "climate:scope_3_emissions", "label": "Scope 3 Emissions", "content": "Value-chain emissions outside direct operations.", "bucket": "metric"},
        {"id": "climate:internal_carbon_price", "label": "Internal Carbon Price", "content": "Shadow price used in planning, capital allocation, or risk management.", "bucket": "framework"},
        {"id": "climate:physical_risk_assessment", "label": "Physical Risk Assessment", "content": "Assessment of hazard, exposure, and vulnerability for assets or portfolios.", "bucket": "framework"},
        {"id": "climate:scenario_analysis", "label": "Scenario Analysis", "content": "Forward-looking stress analysis under alternative climate pathways.", "bucket": "framework"},
        {"id": "climate:financed_emissions", "label": "Financed Emissions", "content": "Emissions associated with lending and investment portfolios.", "bucket": "metric"},
        {"id": "climate:adaptation_plan", "label": "Adaptation Plan", "content": "Operational resilience plan for climate hazards and business continuity.", "bucket": "framework"},
        {"id": "climate:target_coverage", "label": "Target Coverage", "content": "Share of emissions, assets, or operations covered by climate targets.", "bucket": "metric"},
    ],
    "climate:climate_finance": [
        {"id": "climate:resilience_finance", "label": "Resilience Finance", "content": "Capital dedicated to adaptation and resilience projects.", "bucket": "policy"},
        {"id": "climate:green_bonds", "label": "Green Bonds", "content": "Debt instruments earmarked for environmental or climate projects.", "bucket": "policy"},
        {"id": "climate:sustainability_linked_loans", "label": "Sustainability-Linked Loans", "content": "Loans with pricing tied to sustainability performance targets.", "bucket": "policy"},
        {"id": "climate:blended_finance", "label": "Blended Finance", "content": "Capital structure using concessional funding to crowd in private investment.", "bucket": "policy"},
        {"id": "climate:adaptation_finance_gap", "label": "Adaptation Finance Gap", "content": "Difference between needed and deployed adaptation finance.", "bucket": "adaptation"},
        {"id": "climate:just_transition_finance", "label": "Just Transition Finance", "content": "Finance supporting decarbonization while cushioning distributional impacts.", "bucket": "policy"},
        {"id": "climate:multilateral_finance", "label": "Multilateral Finance", "content": "Climate finance provided by development banks and international institutions.", "bucket": "policy"},
        {"id": "climate:private_transition_capex", "label": "Private Transition Capex", "content": "Private investment in decarbonization technologies and infrastructure.", "bucket": "policy"},
        {"id": "climate:cat_bonds", "label": "Catastrophe Bonds", "content": "Insurance-linked securities transferring disaster risk to capital markets.", "bucket": "policy"},
        {"id": "climate:parametric_insurance", "label": "Parametric Insurance", "content": "Insurance paying out based on event parameters rather than audited losses.", "bucket": "policy"},
    ],
    "climate:disclosure_framework": [
        {"id": "climate:tcfd_framework", "label": "TCFD Framework", "content": "Task Force on Climate-related Financial Disclosures framework.", "bucket": "framework"},
        {"id": "climate:issb_ifrs_s2", "label": "ISSB IFRS S2", "content": "International sustainability disclosure standard for climate-related reporting.", "bucket": "framework"},
        {"id": "climate:ghg_protocol", "label": "GHG Protocol", "content": "Accounting standard for greenhouse-gas inventories and reporting.", "bucket": "framework"},
        {"id": "climate:pcaf_standard", "label": "PCAF Standard", "content": "Framework for measuring financed emissions in financial institutions.", "bucket": "framework"},
        {"id": "climate:ngfs_scenarios", "label": "NGFS Scenarios", "content": "Scenario set used for macro-financial climate stress testing.", "bucket": "framework"},
        {"id": "climate:csrd", "label": "CSRD", "content": "EU Corporate Sustainability Reporting Directive.", "bucket": "framework"},
        {"id": "climate:sec_climate_rule", "label": "SEC Climate Disclosure Rule", "content": "US disclosure framework proposal and rulemaking process for climate risk.", "bucket": "framework"},
        {"id": "climate:paris_agreement", "label": "Paris Agreement", "content": "International climate treaty governing NDC-based mitigation coordination.", "bucket": "framework"},
        {"id": "climate:global_stocktake", "label": "Global Stocktake", "content": "Periodic assessment of collective progress under the Paris Agreement.", "bucket": "framework"},
        {"id": "climate:science_based_targets", "label": "Science Based Targets", "content": "Method and validation process for emissions targets aligned with climate goals.", "bucket": "framework"},
    ],
    "climate:metric": [
        {"id": "climate:co2_concentration", "label": "CO2 Concentration", "content": "Atmospheric concentration of carbon dioxide.", "bucket": "metric"},
        {"id": "climate:methane_concentration", "label": "Methane Concentration", "content": "Atmospheric concentration of methane.", "bucket": "metric"},
        {"id": "climate:global_surface_temperature", "label": "Global Surface Temperature", "content": "Observed global surface temperature level or anomaly.", "bucket": "metric"},
        {"id": "climate:ocean_heat_content_anomaly", "label": "Ocean Heat Content Anomaly", "content": "Observed anomaly in ocean heat storage.", "bucket": "metric"},
        {"id": "climate:sea_level_trend", "label": "Sea-Level Trend", "content": "Observed trend in mean sea level.", "bucket": "metric"},
        {"id": "climate:arctic_sea_ice_extent", "label": "Arctic Sea Ice Extent", "content": "Observed Arctic sea ice extent metric.", "bucket": "metric"},
        {"id": "climate:glacier_mass_balance", "label": "Glacier Mass Balance", "content": "Observed glacier mass gain or loss.", "bucket": "metric"},
        {"id": "climate:wildfire_burned_area", "label": "Wildfire Burned Area", "content": "Observed area burned by wildfires.", "bucket": "metric"},
        {"id": "climate:drought_index", "label": "Drought Index", "content": "Standardized drought severity indicator.", "bucket": "metric"},
        {"id": "climate:precipitation_anomaly", "label": "Precipitation Anomaly", "content": "Observed anomaly in precipitation relative to baseline.", "bucket": "metric"},
        {"id": "climate:heatwave_days", "label": "Heatwave Days", "content": "Count of days exceeding an extreme heat threshold.", "bucket": "metric"},
        {"id": "climate:cyclone_accumulated_energy", "label": "Cyclone Accumulated Energy", "content": "Cyclone energy index summarizing storm activity.", "bucket": "metric"},
        {"id": "climate:insured_loss_index", "label": "Insured Loss Index", "content": "Index or aggregate measure of climate-related insured catastrophe losses.", "bucket": "metric"},
        {"id": "climate:emissions_gap_metric", "label": "Emissions Gap", "content": "Gap between projected emissions and target-consistent emissions.", "bucket": "metric"},
        {"id": "climate:renewable_share", "label": "Renewable Share", "content": "Share of energy or electricity supplied by renewables.", "bucket": "metric"},
        {"id": "climate:carbon_intensity", "label": "Carbon Intensity", "content": "Emissions per unit of output, energy, or portfolio exposure.", "bucket": "metric"},
        {"id": "climate:adaptation_gap_metric", "label": "Adaptation Gap Metric", "content": "Observed indicator of resilience shortfall versus required adaptation.", "bucket": "metric"},
        {"id": "climate:climate_finance_flow", "label": "Climate Finance Flow", "content": "Observed flow of climate mitigation and adaptation finance.", "bucket": "metric"},
    ],
    "climate:actor": [
        {"id": "climate:households", "label": "Households", "content": "Households as emitters, voters, and bearers of physical and transition risk.", "bucket": "actor"},
        {"id": "climate:firms", "label": "Firms", "content": "Private-sector firms with emissions, capital stock, and climate exposure.", "bucket": "actor"},
        {"id": "climate:utilities", "label": "Utilities", "content": "Power and water utilities central to transition and resilience.", "bucket": "actor"},
        {"id": "climate:banks", "label": "Banks", "content": "Lenders transmitting climate risk through credit and funding channels.", "bucket": "actor"},
        {"id": "climate:insurers", "label": "Insurers", "content": "Insurers pricing, transferring, and concentrating catastrophe risk.", "bucket": "actor"},
        {"id": "climate:asset_managers", "label": "Asset Managers", "content": "Investors allocating capital across climate-sensitive assets.", "bucket": "actor"},
        {"id": "climate:governments", "label": "Governments", "content": "National governments setting policy, fiscal response, and adaptation strategy.", "bucket": "actor"},
        {"id": "climate:central_banks", "label": "Central Banks", "content": "Authorities concerned with macro-financial stability under climate risk.", "bucket": "actor"},
        {"id": "climate:multilateral_development_banks", "label": "Multilateral Development Banks", "content": "Institutions financing mitigation, adaptation, and recovery.", "bucket": "actor"},
        {"id": "climate:regulators", "label": "Regulators", "content": "Regulators defining disclosure, prudential, and market rules.", "bucket": "actor"},
        {"id": "climate:cities", "label": "Cities", "content": "Subnational actors facing concentrated heat, flood, and infrastructure risk.", "bucket": "actor"},
        {"id": "climate:farmers", "label": "Farmers", "content": "Agricultural producers managing weather, water, and transition exposure.", "bucket": "actor"},
    ],
    "climate:scenario": [
        {"id": "climate:coordinated_transition", "label": "Coordinated Transition", "content": "Orderly policy coordination with rising mitigation and contained macro stress.", "bucket": "scenario"},
        {"id": "climate:fragmented_adaptation", "label": "Fragmented Adaptation", "content": "Uneven resilience buildout with repeated local losses and weak coordination.", "bucket": "scenario"},
        {"id": "climate:climate_crisis", "label": "Climate Crisis", "content": "Persistent hazard escalation with severe macro-social disruption.", "bucket": "scenario"},
        {"id": "climate:policy_backlash", "label": "Policy Backlash", "content": "Political response that weakens climate policy after transition stress or shocks.", "bucket": "scenario"},
        {"id": "climate:delayed_transition", "label": "Delayed Transition", "content": "Late policy tightening after prolonged inaction and accumulated exposure.", "bucket": "scenario"},
        {"id": "climate:disorderly_transition", "label": "Disorderly Transition", "content": "Abrupt decarbonization that generates concentrated transition losses.", "bucket": "scenario"},
        {"id": "climate:resilient_development", "label": "Resilient Development", "content": "Growth path with rising resilience, lower emissions intensity, and lower vulnerability.", "bucket": "scenario"},
        {"id": "climate:chronic_instability", "label": "Chronic Instability", "content": "Repeated shocks that keep economies and institutions in a fragile state.", "bucket": "scenario"},
        {"id": "climate:rapid_decarbonization", "label": "Rapid Decarbonization", "content": "Fast emissions decline from strong policy and technology diffusion.", "bucket": "scenario"},
        {"id": "climate:adaptation_shortfall", "label": "Adaptation Shortfall", "content": "Risk regime where resilience investment persistently lags hazard growth.", "bucket": "scenario"},
    ],
}


def _node_payload(
    node_id: str,
    label: str,
    node_type: str,
    content: str,
    bucket: str,
) -> dict[str, Any]:
    source_keys = BUCKET_SOURCES[bucket]
    return {
        "id": node_id,
        "label": label,
        "node_type": node_type,
        "content": content,
        "confidence": 0.94,
        "status": "active",
        "evidence": [],
        "sources": _source_payload(source_keys),
        "metadata": {
            "seed_version": SEED_VERSION,
            "bucket": bucket,
            "source_keys": source_keys,
        },
        "embedding": [],
        "created_at": SEED_TIMESTAMP,
        "updated_at": SEED_TIMESTAMP,
        "archived_at": None,
    }


def _edge_payload(
    source: str,
    relation_type: str,
    target: str,
    *,
    confidence: float = 0.9,
    weight: float = 1.0,
    bucket: str = "root",
) -> dict[str, Any]:
    edge_id = f"{source}:{relation_type}:{target}"
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation_type": relation_type,
        "confidence": confidence,
        "weight": weight,
        "metadata": {
            "seed_version": SEED_VERSION,
            "bucket": bucket,
            "source_keys": BUCKET_SOURCES[bucket],
        },
        "created_at": SEED_TIMESTAMP,
        "updated_at": SEED_TIMESTAMP,
    }


def build_seed_graph(json_path: str | Path = DEFAULT_SEED_PATH) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(node_id: str, label: str, node_type: str, content: str, bucket: str) -> None:
        if node_id in nodes:
            raise ValueError(f"Duplicate node id: {node_id}")
        nodes[node_id] = _node_payload(node_id, label, node_type, content, bucket)

    def add_edge(source: str, relation_type: str, target: str, *, confidence: float = 0.9, weight: float = 1.0, bucket: str = "root") -> None:
        if source not in nodes or target not in nodes:
            raise KeyError(f"Unknown node in edge: {source} -[{relation_type}]-> {target}")
        edge = _edge_payload(source, relation_type, target, confidence=confidence, weight=weight, bucket=bucket)
        if edge["id"] in edges:
            return
        edges[edge["id"]] = edge

    for node in CORE_NODES:
        add_node(node["id"], node["label"], node["node_type"], node["content"], node["bucket"])

    for node in CORE_NODES:
        if node.get("parent"):
            add_edge(node["id"], "type_of", node["parent"], confidence=0.98, bucket=node["bucket"])

    for parent_id, children in CHILDREN_BY_PARENT.items():
        parent_bucket = nodes[parent_id]["metadata"]["bucket"]
        for child in children:
            inferred_type = "concept"
            if child["bucket"] in {"metric"}:
                inferred_type = "metric"
            elif child["bucket"] in {"policy", "adaptation", "framework"}:
                inferred_type = "policy"
            elif child["bucket"] in {"finance", "impact"}:
                inferred_type = "risk"
            elif child["bucket"] in {"hazard"}:
                inferred_type = "hazard"
            elif child["bucket"] in {"actor"}:
                inferred_type = "actor"
            elif child["bucket"] in {"scenario"}:
                inferred_type = "scenario"
            elif child["bucket"] in {"technology"}:
                inferred_type = "technology"
            add_node(child["id"], child["label"], inferred_type, child["content"], child["bucket"])
            add_edge(child["id"], "type_of", parent_id, confidence=0.97, bucket=child["bucket"] if child["bucket"] in BUCKET_SOURCES else parent_bucket)

    for forcing_node in [
        "climate:carbon_dioxide",
        "climate:methane",
        "climate:nitrous_oxide",
        "climate:fluorinated_gases",
        "climate:black_carbon",
        "climate:tropospheric_ozone",
        "climate:land_use_change",
        "climate:cement_process_emissions",
    ]:
        add_edge(forcing_node, "causes", "climate:anthropogenic_emissions", bucket="forcing")
        add_edge(forcing_node, "causes", "climate:radiative_forcing", bucket="forcing")

    add_edge("climate:anthropogenic_emissions", "causes", "climate:radiative_forcing", bucket="forcing")
    add_edge("climate:radiative_forcing", "causes", "climate:earth_energy_imbalance", bucket="forcing")
    add_edge("climate:earth_energy_imbalance", "causes", "climate:global_warming", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:ocean_heat_content", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:cryosphere_loss", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:hydrological_intensification", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:mean_temperature_anomaly", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:sea_level_rise", bucket="response")
    add_edge("climate:global_warming", "causes", "climate:ecosystem_regime_shift", bucket="response")
    add_edge("climate:cryosphere_loss", "causes", "climate:sea_level_rise", bucket="response")
    add_edge("climate:glacier_mass_loss", "causes", "climate:sea_level_rise", bucket="response")
    add_edge("climate:ice_sheet_instability", "causes", "climate:sea_level_rise", bucket="response")
    add_edge("climate:permafrost_thaw", "causes", "climate:carbon_cycle_feedback", bucket="response")
    add_edge("climate:carbon_cycle_feedback", "causes", "climate:radiative_forcing", bucket="response")
    add_edge("climate:arctic_sea_ice_decline", "causes", "climate:albedo_change", bucket="response")
    add_edge("climate:albedo_change", "causes", "climate:earth_energy_imbalance", bucket="response")
    add_edge("climate:carbon_dioxide", "causes", "climate:ocean_acidification", bucket="response")
    add_edge("climate:aerosol_masking", "reduces", "climate:global_warming", bucket="forcing")

    for hazard_id in [
        "climate:heatwave",
        "climate:marine_heatwave",
        "climate:drought",
        "climate:flash_drought",
        "climate:wildfire_weather",
        "climate:heavy_precipitation",
        "climate:river_flood",
        "climate:pluvial_flood",
        "climate:coastal_flood",
        "climate:storm_surge",
        "climate:tropical_cyclone_intensity",
        "climate:tropical_cyclone_rainfall",
        "climate:compound_event",
        "climate:crop_heat_stress",
        "climate:water_scarcity",
        "climate:vector_borne_disease_risk",
        "climate:coral_bleaching",
        "climate:landslide_risk",
    ]:
        add_edge("climate:global_warming", "causes", hazard_id, bucket="hazard")

    for hazard_id in [
        "climate:heavy_precipitation",
        "climate:river_flood",
        "climate:pluvial_flood",
        "climate:landslide_risk",
        "climate:tropical_cyclone_rainfall",
    ]:
        add_edge("climate:hydrological_intensification", "causes", hazard_id, bucket="hazard")

    for hazard_id in ["climate:coastal_flood", "climate:storm_surge"]:
        add_edge("climate:sea_level_rise", "causes", hazard_id, bucket="hazard")

    exposure_map = {
        "climate:heatwave": ["climate:urban_heat_burden", "climate:labor_exposure", "climate:health_system_exposure", "climate:grid_exposure"],
        "climate:marine_heatwave": ["climate:ecosystem_service_exposure", "climate:tourism_exposure"],
        "climate:drought": ["climate:water_stress", "climate:agricultural_exposure", "climate:food_system_exposure", "climate:forest_exposure"],
        "climate:flash_drought": ["climate:water_stress", "climate:agricultural_exposure"],
        "climate:wildfire_weather": ["climate:forest_exposure", "climate:housing_exposure", "climate:health_system_exposure", "climate:infrastructure_exposure"],
        "climate:heavy_precipitation": ["climate:infrastructure_exposure", "climate:housing_exposure", "climate:port_exposure"],
        "climate:river_flood": ["climate:infrastructure_exposure", "climate:housing_exposure", "climate:food_system_exposure"],
        "climate:pluvial_flood": ["climate:urban_heat_burden", "climate:infrastructure_exposure", "climate:housing_exposure"],
        "climate:coastal_flood": ["climate:coastal_asset_exposure", "climate:port_exposure", "climate:housing_exposure"],
        "climate:storm_surge": ["climate:coastal_asset_exposure", "climate:port_exposure"],
        "climate:tropical_cyclone_intensity": ["climate:coastal_asset_exposure", "climate:grid_exposure", "climate:infrastructure_exposure"],
        "climate:tropical_cyclone_rainfall": ["climate:port_exposure", "climate:supply_chain_exposure", "climate:housing_exposure"],
        "climate:compound_event": ["climate:supply_chain_exposure", "climate:food_system_exposure", "climate:coastal_asset_exposure"],
        "climate:crop_heat_stress": ["climate:agricultural_exposure", "climate:food_system_exposure"],
        "climate:water_scarcity": ["climate:water_stress", "climate:water_quality_risk"],
        "climate:vector_borne_disease_risk": ["climate:health_system_exposure"],
        "climate:coral_bleaching": ["climate:ecosystem_service_exposure", "climate:tourism_exposure"],
        "climate:landslide_risk": ["climate:infrastructure_exposure", "climate:housing_exposure"],
    }
    for hazard_id, exposure_ids in exposure_map.items():
        for exposure_id in exposure_ids:
            add_edge(hazard_id, "impacts", exposure_id, bucket="exposure")

    impact_map = {
        "climate:coastal_asset_exposure": ["climate:coastal_inundation_risk", "climate:infrastructure_damage_risk"],
        "climate:urban_heat_burden": ["climate:chronic_heat_risk", "climate:health_risk"],
        "climate:water_stress": ["climate:water_supply_risk", "climate:food_security_risk"],
        "climate:food_system_exposure": ["climate:crop_yield_risk", "climate:food_security_risk"],
        "climate:infrastructure_exposure": ["climate:infrastructure_damage_risk", "climate:supply_chain_risk"],
        "climate:health_system_exposure": ["climate:health_risk"],
        "climate:labor_exposure": ["climate:chronic_heat_risk"],
        "climate:housing_exposure": ["climate:acute_flood_risk", "climate:wildfire_risk"],
        "climate:supply_chain_exposure": ["climate:supply_chain_risk"],
        "climate:ecosystem_service_exposure": ["climate:ecosystem_loss_risk"],
        "climate:tourism_exposure": ["climate:productivity_loss"],
        "climate:insurance_penetration_gap": ["climate:uninsured_loss"],
        "climate:grid_exposure": ["climate:power_system_risk"],
        "climate:agricultural_exposure": ["climate:crop_yield_risk", "climate:food_security_risk"],
        "climate:forest_exposure": ["climate:wildfire_risk", "climate:ecosystem_loss_risk"],
        "climate:port_exposure": ["climate:supply_chain_risk", "climate:trade_disruption"],
    }
    for exposure_id, impact_ids in impact_map.items():
        for impact_id in impact_ids:
            add_edge(exposure_id, "impacts", impact_id, bucket="impact")

    risk_to_loss = {
        "climate:chronic_heat_risk": ["climate:productivity_loss", "climate:health_risk"],
        "climate:acute_flood_risk": ["climate:asset_impairment", "climate:insured_loss", "climate:uninsured_loss"],
        "climate:wildfire_risk": ["climate:insured_loss", "climate:uninsured_loss", "climate:health_risk"],
        "climate:water_supply_risk": ["climate:gdp_loss", "climate:commodity_price_spike"],
        "climate:crop_yield_risk": ["climate:gdp_loss", "climate:commodity_price_spike", "climate:inflation_pressure"],
        "climate:health_risk": ["climate:productivity_loss", "climate:fiscal_pressure"],
        "climate:infrastructure_damage_risk": ["climate:asset_impairment", "climate:trade_disruption"],
        "climate:ecosystem_loss_risk": ["climate:gdp_loss", "climate:adaptation_gap"],
        "climate:coastal_inundation_risk": ["climate:asset_impairment", "climate:migration_risk"],
        "climate:power_system_risk": ["climate:gdp_loss", "climate:inflation_pressure"],
        "climate:migration_risk": ["climate:fiscal_pressure", "climate:chronic_instability"],
        "climate:supply_chain_risk": ["climate:trade_disruption", "climate:inflation_pressure"],
        "climate:food_security_risk": ["climate:commodity_price_spike", "climate:fiscal_pressure"],
        "climate:water_quality_risk": ["climate:health_risk", "climate:fiscal_pressure"],
    }
    for risk_id, loss_ids in risk_to_loss.items():
        for loss_id in loss_ids:
            if loss_id.startswith("climate:") and loss_id in nodes:
                add_edge(risk_id, "impacts", loss_id, bucket="impact")

    financial_map = {
        "climate:gdp_loss": ["climate:credit_risk", "climate:sovereign_spread_widening", "climate:macro_financial_feedback"],
        "climate:asset_impairment": ["climate:collateral_repricing", "climate:mortgage_default_risk"],
        "climate:productivity_loss": ["climate:credit_risk", "climate:market_repricing"],
        "climate:fiscal_pressure": ["climate:sovereign_spread_widening", "climate:fiscal_financial_linkage"],
        "climate:inflation_pressure": ["climate:liquidity_stress", "climate:market_repricing"],
        "climate:trade_disruption": ["climate:credit_risk", "climate:market_repricing"],
        "climate:commodity_price_spike": ["climate:market_repricing", "climate:liquidity_stress"],
        "climate:insured_loss": ["climate:insurance_losses", "climate:reserve_adequacy_pressure"],
        "climate:uninsured_loss": ["climate:mortgage_default_risk", "climate:municipal_bond_stress"],
        "climate:adaptation_gap": ["climate:insurance_losses", "climate:solvency_pressure"],
    }
    for loss_id, finance_ids in financial_map.items():
        for finance_id in finance_ids:
            add_edge(loss_id, "impacts", finance_id, bucket="finance")

    for finance_id in [
        "climate:insurance_losses",
        "climate:mortgage_default_risk",
        "climate:credit_risk",
        "climate:market_repricing",
        "climate:liquidity_stress",
        "climate:sovereign_spread_widening",
        "climate:municipal_bond_stress",
        "climate:reserve_adequacy_pressure",
        "climate:collateral_repricing",
        "climate:solvency_pressure",
        "climate:macro_financial_feedback",
        "climate:fiscal_financial_linkage",
    ]:
        add_edge(finance_id, "impacts", "climate:financial_stability", bucket="finance")

    for transition_risk_id in [
        "climate:policy_shock_risk",
        "climate:technology_disruption_risk",
        "climate:reputation_risk",
        "climate:litigation_risk",
        "climate:stranded_asset_risk",
        "climate:carbon_cost_risk",
        "climate:demand_shift_risk",
        "climate:supply_side_transition_risk",
        "climate:disclosure_gap_risk",
        "climate:transition_plan_gap",
    ]:
        add_edge(transition_risk_id, "impacts", "climate:financial_stability", bucket="finance")

    policy_edges = [
        ("climate:mitigation_policy", "reduces", "climate:anthropogenic_emissions"),
        ("climate:carbon_pricing", "type_of", "climate:mitigation_policy"),
        ("climate:carbon_pricing", "reduces", "climate:anthropogenic_emissions"),
        ("climate:emissions_standard", "reduces", "climate:anthropogenic_emissions"),
        ("climate:methane_regulation", "reduces", "climate:methane"),
        ("climate:coal_phaseout", "reduces", "climate:carbon_dioxide"),
        ("climate:renewable_subsidy", "enables", "climate:solar_pv"),
        ("climate:renewable_subsidy", "enables", "climate:wind_power"),
        ("climate:clean_power_standard", "enables", "climate:renewable_share"),
        ("climate:ev_mandate", "enables", "climate:heat_pumps"),
        ("climate:building_efficiency_code", "enables", "climate:building_retrofits"),
        ("climate:deforestation_control", "reduces", "climate:land_use_change"),
        ("climate:industrial_decarbonization_policy", "enables", "climate:low_carbon_cement"),
        ("climate:industrial_decarbonization_policy", "enables", "climate:low_carbon_steel"),
        ("climate:ndc_ambition", "reduces", "climate:net_zero_gap"),
        ("climate:loss_and_damage_fund", "enables", "climate:resilience_finance"),
        ("climate:climate_stress_testing", "governs", "climate:financial_stability"),
        ("climate:green_taxonomy", "governs", "climate:climate_finance"),
        ("climate:adaptation_investment", "reduces", "climate:adaptation_gap"),
        ("climate:adaptation_investment", "reduces", "climate:physical_risk"),
        ("climate:early_warning_system", "reduces", "climate:health_risk"),
        ("climate:early_warning_system", "reduces", "climate:acute_flood_risk"),
        ("climate:flood_defense", "reduces", "climate:acute_flood_risk"),
        ("climate:flood_defense", "reduces", "climate:coastal_inundation_risk"),
        ("climate:drought_resilient_crops", "reduces", "climate:crop_yield_risk"),
        ("climate:wildfire_fuel_management", "reduces", "climate:wildfire_risk"),
        ("climate:urban_cooling", "reduces", "climate:chronic_heat_risk"),
        ("climate:water_reuse", "reduces", "climate:water_supply_risk"),
        ("climate:desalination", "reduces", "climate:water_supply_risk"),
        ("climate:resilient_grid", "reduces", "climate:power_system_risk"),
        ("climate:managed_retreat", "reduces", "climate:coastal_asset_exposure"),
        ("climate:resilient_building_codes", "reduces", "climate:infrastructure_damage_risk"),
        ("climate:catastrophe_insurance", "reduces", "climate:uninsured_loss"),
        ("climate:nature_based_solutions", "reduces", "climate:coastal_inundation_risk"),
        ("climate:heat_health_action_plan", "reduces", "climate:health_risk"),
        ("climate:irrigation_efficiency", "reduces", "climate:water_stress"),
        ("climate:climate_services", "reduces", "climate:adaptation_gap"),
        ("climate:solar_pv", "reduces", "climate:carbon_intensity"),
        ("climate:wind_power", "reduces", "climate:carbon_intensity"),
        ("climate:battery_storage", "enables", "climate:renewable_share"),
        ("climate:grid_transmission", "enables", "climate:renewable_share"),
        ("climate:heat_pumps", "reduces", "climate:carbon_intensity"),
        ("climate:green_hydrogen", "reduces", "climate:carbon_intensity"),
        ("climate:nuclear_power", "reduces", "climate:carbon_intensity"),
        ("climate:carbon_capture_storage", "reduces", "climate:carbon_dioxide"),
        ("climate:direct_air_capture", "reduces", "climate:co2_concentration"),
        ("climate:demand_response", "enables", "climate:resilient_grid"),
        ("climate:building_retrofits", "reduces", "climate:carbon_intensity"),
        ("climate:low_carbon_cement", "reduces", "climate:cement_process_emissions"),
        ("climate:low_carbon_steel", "reduces", "climate:carbon_intensity"),
        ("climate:sustainable_aviation_fuel", "reduces", "climate:carbon_intensity"),
        ("climate:tcfd_framework", "governs", "climate:climate_disclosure"),
        ("climate:issb_ifrs_s2", "governs", "climate:climate_disclosure"),
        ("climate:ghg_protocol", "governs", "climate:scope_1_emissions"),
        ("climate:ghg_protocol", "governs", "climate:scope_2_emissions"),
        ("climate:ghg_protocol", "governs", "climate:scope_3_emissions"),
        ("climate:pcaf_standard", "governs", "climate:financed_emissions"),
        ("climate:ngfs_scenarios", "governs", "climate:scenario_analysis"),
        ("climate:csrd", "governs", "climate:climate_disclosure"),
        ("climate:sec_climate_rule", "governs", "climate:climate_disclosure"),
        ("climate:paris_agreement", "governs", "climate:ndc_ambition"),
        ("climate:global_stocktake", "signals", "climate:net_zero_gap"),
        ("climate:science_based_targets", "governs", "climate:transition_plan"),
        ("climate:climate_disclosure", "reduces", "climate:disclosure_gap_risk"),
        ("climate:transition_plan", "reduces", "climate:transition_plan_gap"),
        ("climate:scenario_analysis", "reduces", "climate:disclosure_gap_risk"),
        ("climate:physical_risk_assessment", "reduces", "climate:disclosure_gap_risk"),
        ("climate:internal_carbon_price", "reduces", "climate:carbon_cost_risk"),
        ("climate:adaptation_plan", "reduces", "climate:adaptation_gap"),
        ("climate:resilience_finance", "enables", "climate:adaptation_investment"),
        ("climate:green_bonds", "enables", "climate:private_transition_capex"),
        ("climate:sustainability_linked_loans", "enables", "climate:private_transition_capex"),
        ("climate:blended_finance", "enables", "climate:resilience_finance"),
        ("climate:adaptation_finance_gap", "impacts", "climate:adaptation_gap"),
        ("climate:just_transition_finance", "reduces", "climate:policy_backlash"),
        ("climate:multilateral_finance", "enables", "climate:resilience_finance"),
        ("climate:private_transition_capex", "enables", "climate:rapid_decarbonization"),
        ("climate:cat_bonds", "reduces", "climate:insurance_losses"),
        ("climate:parametric_insurance", "reduces", "climate:uninsured_loss"),
    ]
    for source, relation, target in policy_edges:
        add_edge(source, relation, target, bucket="policy" if relation in {"enables", "governs"} else "adaptation" if source.startswith("climate:adaptation") else "policy")

    metric_edges = [
        ("climate:co2_concentration", "measures", "climate:carbon_dioxide"),
        ("climate:methane_concentration", "measures", "climate:methane"),
        ("climate:global_surface_temperature", "measures", "climate:global_warming"),
        ("climate:ocean_heat_content_anomaly", "measures", "climate:ocean_heat_content"),
        ("climate:sea_level_trend", "measures", "climate:sea_level_rise"),
        ("climate:arctic_sea_ice_extent", "measures", "climate:arctic_sea_ice_decline"),
        ("climate:glacier_mass_balance", "measures", "climate:glacier_mass_loss"),
        ("climate:wildfire_burned_area", "measures", "climate:wildfire_risk"),
        ("climate:drought_index", "measures", "climate:drought"),
        ("climate:precipitation_anomaly", "measures", "climate:hydrological_intensification"),
        ("climate:heatwave_days", "measures", "climate:heatwave"),
        ("climate:cyclone_accumulated_energy", "measures", "climate:tropical_cyclone_intensity"),
        ("climate:insured_loss_index", "measures", "climate:insurance_losses"),
        ("climate:emissions_gap_metric", "measures", "climate:net_zero_gap"),
        ("climate:renewable_share", "measures", "climate:rapid_decarbonization"),
        ("climate:carbon_intensity", "measures", "climate:anthropogenic_emissions"),
        ("climate:adaptation_gap_metric", "measures", "climate:adaptation_gap"),
        ("climate:climate_finance_flow", "measures", "climate:climate_finance"),
        ("climate:scope_1_emissions", "measures", "climate:anthropogenic_emissions"),
        ("climate:scope_2_emissions", "measures", "climate:carbon_intensity"),
        ("climate:scope_3_emissions", "measures", "climate:transition_plan_gap"),
        ("climate:financed_emissions", "measures", "climate:transition_risk"),
        ("climate:target_coverage", "measures", "climate:transition_plan"),
        ("climate:mean_temperature_anomaly", "measures", "climate:global_warming"),
        ("climate:marine_heat_content_anomaly", "measures", "climate:ocean_heat_content"),
    ]
    for source, relation, target in metric_edges:
        add_edge(source, relation, target, bucket="metric")

    actor_edges = [
        ("climate:governments", "governs", "climate:mitigation_policy"),
        ("climate:governments", "governs", "climate:adaptation_investment"),
        ("climate:governments", "governs", "climate:loss_and_damage_fund"),
        ("climate:regulators", "governs", "climate:climate_disclosure"),
        ("climate:regulators", "governs", "climate:climate_stress_testing"),
        ("climate:central_banks", "governs", "climate:financial_stability"),
        ("climate:multilateral_development_banks", "enables", "climate:multilateral_finance"),
        ("climate:banks", "enables", "climate:sustainability_linked_loans"),
        ("climate:insurers", "enables", "climate:catastrophe_insurance"),
        ("climate:asset_managers", "enables", "climate:green_bonds"),
        ("climate:utilities", "enables", "climate:grid_transmission"),
        ("climate:cities", "enables", "climate:urban_cooling"),
        ("climate:farmers", "enables", "climate:drought_resilient_crops"),
        ("climate:firms", "enables", "climate:transition_plan"),
        ("climate:households", "impacts", "climate:policy_backlash"),
    ]
    for source, relation, target in actor_edges:
        add_edge(source, relation, target, bucket="actor")

    scenario_edges = [
        ("climate:mitigation_policy", "supports", "climate:coordinated_transition"),
        ("climate:adaptation_investment", "supports", "climate:fragmented_adaptation"),
        ("climate:adaptation_investment", "supports", "climate:resilient_development"),
        ("climate:global_warming", "supports", "climate:climate_crisis"),
        ("climate:physical_risk", "supports", "climate:climate_crisis"),
        ("climate:economic_loss", "supports", "climate:chronic_instability"),
        ("climate:financial_stability", "supports", "climate:coordinated_transition"),
        ("climate:transition_risk", "supports", "climate:disorderly_transition"),
        ("climate:policy_backlash", "supports", "climate:delayed_transition"),
        ("climate:net_zero_gap", "supports", "climate:delayed_transition"),
        ("climate:rapid_decarbonization", "supports", "climate:coordinated_transition"),
        ("climate:adaptation_gap", "supports", "climate:adaptation_shortfall"),
        ("climate:fragmented_adaptation", "impacts", "climate:financial_stability"),
        ("climate:climate_crisis", "impacts", "climate:financial_stability"),
    ]
    for source, relation, target in scenario_edges:
        add_edge(source, relation, target, bucket="scenario")

    return {
        "backend": "networkx-json",
        "json_path": str(Path(json_path)),
        "seed_version": SEED_VERSION,
        "source_references": SOURCE_REFERENCES,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def write_seed_graph(
    *,
    seed_path: str | Path = DEFAULT_SEED_PATH,
    memory_path: str | Path | None = None,
    backup_existing: bool = True,
) -> tuple[Path, Path | None]:
    seed_target = Path(seed_path)
    payload = build_seed_graph(seed_target)
    seed_target.parent.mkdir(parents=True, exist_ok=True)
    seed_target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    memory_target: Path | None = None
    if memory_path is not None:
        memory_target = Path(memory_path)
        memory_target.parent.mkdir(parents=True, exist_ok=True)
        if memory_target.exists() and backup_existing:
            backup_path = memory_target.with_suffix(memory_target.suffix + ".preseed.bak")
            shutil.copy2(memory_target, backup_path)
        memory_payload = dict(payload)
        memory_payload["json_path"] = str(memory_target)
        memory_target.write_text(json.dumps(memory_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return seed_target, memory_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Freeman climate seed knowledge graph.")
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH), help="Where to write the immutable seed graph JSON.")
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH), help="Where to write the active memory graph JSON.")
    parser.add_argument(
        "--write-memory",
        action="store_true",
        help="Also write the generated graph to the active memory path used by config.climate.yaml.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create *.preseed.bak before overwriting an existing memory graph.",
    )
    args = parser.parse_args()

    seed_target, memory_target = write_seed_graph(
        seed_path=args.seed_path,
        memory_path=args.memory_path if args.write_memory else None,
        backup_existing=not args.no_backup,
    )
    payload = json.loads(seed_target.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "seed_path": str(seed_target),
                "memory_path": str(memory_target) if memory_target is not None else None,
                "seed_version": payload["seed_version"],
                "node_count": len(payload["nodes"]),
                "edge_count": len(payload["edges"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
