"""Tests for the geospatial regional adapter."""

from __future__ import annotations

import pytest

from freeman.domain.compiler import DomainCompiler
from freeman.domain.schema import GEO_BORDERED_BY, GEO_WITHIN_REGION, SPATIAL_REGION_NODE_TYPE
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.realworld.spatial_adapter import SpatialAdapter

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")


def _regional_gdf():
    return gpd.GeoDataFrame(
        [
            {"geo_id": "A", "name": "Region A", "climate_zone": "arid"},
            {"geo_id": "B", "name": "Region B", "climate_zone": "temperate"},
            {"geo_id": "C", "name": "Region C", "climate_zone": "arid"},
        ],
        geometry=[
            shapely_geometry.box(0.0, 0.0, 2.0, 2.0),
            shapely_geometry.box(2.0, 0.0, 4.0, 2.0),
            shapely_geometry.box(0.5, 0.5, 1.0, 1.0),
        ],
        crs="EPSG:4326",
    )


def test_spatial_adapter_exports_regions_and_topology_to_kg(tmp_path) -> None:
    adapter = SpatialAdapter(_regional_gdf(), geo_id_column="geo_id", label_column="name")

    assert adapter.region_ids == ["A", "B", "C"]
    assert [region.geo_id for region in adapter.touches("A")] == ["B"]
    assert [region.geo_id for region in adapter.within("C")] == ["A"]

    kg = KnowledgeGraph(json_path=tmp_path / "kg.json", auto_load=False, auto_save=False)
    summary = adapter.ingest_into_kg(kg, predicates=("touches", "within"))

    assert summary == {"regions": 3, "nodes": 3, "edges": 2}
    node = kg.get_node("geo:A")
    assert node is not None
    assert node.node_type == SPATIAL_REGION_NODE_TYPE
    assert node.metadata["geo"]["geo:hasGeometry"] == "geometry:A"
    assert node.metadata["geo"]["geo:hasClimateZone"] == "arid"

    edges = sorted((edge.source, edge.target, edge.relation_type) for edge in kg.edges())
    assert ("geo:A", "geo:B", GEO_BORDERED_BY) in edges
    assert ("geo:C", "geo:A", GEO_WITHIN_REGION) in edges


def test_spatial_adapter_loads_geojson_payload() -> None:
    adapter = SpatialAdapter.from_geojson(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "X",
                    "properties": {"name": "Region X", "climate_zone": "polar"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]],
                    },
                }
            ],
        },
        label_column="name",
    )

    assert adapter.region_ids == ["X"]
    assert adapter.region("X").label == "Region X"
    assert adapter.as_kg_nodes()[0].metadata["geo"]["geo:hasClimateZone"] == "polar"


def test_domain_compiler_preserves_spatial_and_geo_metadata(water_market_schema) -> None:
    water_market_schema["spatial"] = {"region_ids": ["A", "B"], "predicate": "geo:borderedBy"}
    water_market_schema["geo"] = {"default_crs": "EPSG:4326"}

    world = DomainCompiler().compile(water_market_schema)

    assert world.metadata["spatial"]["region_ids"] == ["A", "B"]
    assert world.metadata["geo"]["default_crs"] == "EPSG:4326"
