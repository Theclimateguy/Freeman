"""Geospatial adapter for regional analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from freeman.domain.schema import (
    GEO_BORDERED_BY,
    GEO_HAS_CLIMATE_ZONE,
    GEO_HAS_GEOMETRY,
    GEO_INTERSECTS,
    GEO_PREDICATES,
    GEO_WITHIN_REGION,
    SPATIAL_REGION_NODE_TYPE,
)
from freeman.memory.knowledgegraph import KGEdge, KGNode, KnowledgeGraph
from freeman.utils import deep_copy_jsonable, json_ready

_INTERNAL_GEO_ID = "_freeman_geo_id"
_INTERNAL_GEOMETRY_KEY = "_freeman_geometry_key"
_DEFAULT_GEO_ID_COLUMNS = ("geo_id", "id", "ISO_A3", "iso_a3", "ADM0_A3", "GID_0", "name", "NAME")
_DEFAULT_LABEL_COLUMNS = ("name", "NAME", "label", "Label", "admin", "ADMIN")
_PREDICATE_TO_RELATION = {
    "touches": GEO_BORDERED_BY,
    "bordered_by": GEO_BORDERED_BY,
    "borderedBy": GEO_BORDERED_BY,
    GEO_BORDERED_BY: GEO_BORDERED_BY,
    "within": GEO_WITHIN_REGION,
    "within_region": GEO_WITHIN_REGION,
    "withinRegion": GEO_WITHIN_REGION,
    GEO_WITHIN_REGION: GEO_WITHIN_REGION,
    "intersects": GEO_INTERSECTS,
    GEO_INTERSECTS: GEO_INTERSECTS,
}
_SYMMETRIC_PREDICATES = {"touches", "intersects"}


class GeoDependencyError(ImportError):
    """Raised when optional geospatial dependencies are unavailable."""


def _require_geopandas() -> Any:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise GeoDependencyError('Install geospatial support with `pip install "freeman[geo]"`.') from exc
    return gpd


def _require_shape() -> Any:
    try:
        from shapely.geometry import shape
    except ImportError as exc:
        raise GeoDependencyError('Install geospatial support with `pip install "freeman[geo]"`.') from exc
    return shape


def _clean_geo_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _as_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _json_properties(row: Any, columns: Sequence[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in columns:
        if column in {"geometry", _INTERNAL_GEO_ID, _INTERNAL_GEOMETRY_KEY}:
            continue
        payload[str(column)] = row[column]
    return json_ready(payload)


def _geometry_area(geometry: Any) -> float:
    if geometry is None or getattr(geometry, "is_empty", False):
        return 0.0
    return float(getattr(geometry, "area", 0.0))


def _geometry_centroid(geometry: Any) -> tuple[float, float]:
    if geometry is None or getattr(geometry, "is_empty", False):
        return (0.0, 0.0)
    centroid = geometry.centroid
    return (float(centroid.x), float(centroid.y))


@dataclass(frozen=True)
class SpatialRegion:
    """One regional feature with a stable Freeman geo id."""

    geo_id: str
    label: str
    geometry_key: str
    bounds: tuple[float, float, float, float]
    centroid: tuple[float, float]
    area: float
    properties: dict[str, Any] = field(default_factory=dict)
    crs: str | None = None

    @property
    def node_id(self) -> str:
        return f"geo:{self.geo_id}"

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable spatial metadata without embedding geometry."""

        payload = {
            "geo_id": self.geo_id,
            "label": self.label,
            "geometry_key": self.geometry_key,
            GEO_HAS_GEOMETRY: self.geometry_key,
            "bounds": list(self.bounds),
            "centroid": {"x": self.centroid[0], "y": self.centroid[1]},
            "area": self.area,
            "crs": self.crs,
            "properties": deep_copy_jsonable(self.properties),
        }
        climate_zone = self.properties.get("climate_zone") or self.properties.get("climateZone")
        if climate_zone is not None:
            payload[GEO_HAS_CLIMATE_ZONE] = climate_zone
        return json_ready(payload)

    def to_kg_node(self, *, confidence: float = 0.95, node_prefix: str = "geo:") -> KGNode:
        """Represent the region as a KG node with an external geometry reference."""

        node_id = f"{node_prefix}{self.geo_id}"
        label = self.label or self.geo_id
        return KGNode(
            id=node_id,
            label=label,
            node_type=SPATIAL_REGION_NODE_TYPE,
            content=f"Spatial region {label} ({self.geo_id}).",
            confidence=confidence,
            metadata={"geo": self.snapshot()},
        )


@dataclass(frozen=True)
class SpatialRelation:
    """One topological relation between two spatial regions."""

    source_geo_id: str
    target_geo_id: str
    predicate: str
    relation_type: str
    confidence: float = 0.85
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_kg_edge(self, *, node_prefix: str = "geo:") -> KGEdge:
        """Represent the topological relation as a KG edge."""

        source = f"{node_prefix}{self.source_geo_id}"
        target = f"{node_prefix}{self.target_geo_id}"
        return KGEdge(
            id=f"{source}:{self.relation_type}:{target}",
            source=source,
            target=target,
            relation_type=self.relation_type,
            confidence=self.confidence,
            weight=self.weight,
            metadata=json_ready(
                {
                    "predicate": self.predicate,
                    "source_geo_id": self.source_geo_id,
                    "target_geo_id": self.target_geo_id,
                    **self.metadata,
                }
            ),
        )


class SpatialAdapter:
    """Thin GeoPandas/Shapely adapter for regional Freeman workflows."""

    def __init__(
        self,
        geodataframe: Any,
        *,
        geo_id_column: str | None = None,
        label_column: str | None = None,
        source: str | None = None,
        geometry_key_prefix: str = "geometry",
    ) -> None:
        if not hasattr(geodataframe, "geometry"):
            raise TypeError("SpatialAdapter expects a GeoPandas GeoDataFrame-like object with a geometry column.")
        self.gdf = geodataframe.copy()
        self.source = source
        self.geometry_key_prefix = str(geometry_key_prefix)
        self.crs = str(self.gdf.crs) if getattr(self.gdf, "crs", None) is not None else None
        self.geo_id_column = self._resolve_geo_id_column(geo_id_column)
        self.label_column = self._resolve_label_column(label_column)
        self._prepare_geo_ids()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        geo_id_column: str | None = None,
        label_column: str | None = None,
        crs: str | None = None,
        target_crs: str | None = None,
    ) -> "SpatialAdapter":
        """Load GeoJSON/Shapefile-compatible vector data into the adapter."""

        gpd = _require_geopandas()
        source_path = Path(path).expanduser().resolve()
        gdf = gpd.read_file(source_path)
        if crs is not None and getattr(gdf, "crs", None) is None:
            gdf = gdf.set_crs(crs)
        if target_crs is not None:
            gdf = gdf.to_crs(target_crs)
        return cls(gdf, geo_id_column=geo_id_column, label_column=label_column, source=str(source_path))

    @classmethod
    def from_geojson(
        cls,
        payload: str | bytes | Mapping[str, Any],
        *,
        geo_id_column: str | None = None,
        label_column: str | None = None,
        crs: str | None = "EPSG:4326",
    ) -> "SpatialAdapter":
        """Build an adapter from an in-memory GeoJSON FeatureCollection."""

        gpd = _require_geopandas()
        shape = _require_shape()
        data = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload) if isinstance(
            payload, (str, bytes)
        ) else dict(payload)
        features = data.get("features", [])
        records: list[dict[str, Any]] = []
        geometries: list[Any] = []
        for idx, feature in enumerate(features):
            properties = dict(feature.get("properties", {}))
            properties.setdefault("geo_id", feature.get("id", f"region:{idx}"))
            records.append(properties)
            geometries.append(shape(feature["geometry"]))
        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
        return cls(gdf, geo_id_column=geo_id_column, label_column=label_column, source="geojson")

    @property
    def region_ids(self) -> list[str]:
        """Return adapter geo ids in GeoDataFrame order."""

        return [str(value) for value in self.gdf[_INTERNAL_GEO_ID].tolist()]

    def regions(self) -> list[SpatialRegion]:
        """Return all regions as JSON-safe descriptors."""

        return [self._row_to_region(row) for _, row in self.gdf.iterrows()]

    def region(self, geo_id: str) -> SpatialRegion:
        """Return one region by geo id."""

        return self._row_to_region(self._row_for_geo_id(geo_id))

    def geometry(self, geo_id: str) -> Any:
        """Return the raw Shapely geometry for local topological analysis."""

        return self._row_for_geo_id(geo_id).geometry

    def neighbors(self, geo_id: str, *, predicate: str = "touches") -> list[SpatialRegion]:
        """Return regions satisfying a Shapely topological predicate with ``geo_id``."""

        normalized = self._normalize_predicate(predicate)
        source_geometry = self.geometry(geo_id)
        results: list[SpatialRegion] = []
        for _, row in self.gdf.iterrows():
            candidate_id = str(row[_INTERNAL_GEO_ID])
            if candidate_id == str(geo_id):
                continue
            if self._predicate_matches(normalized, source_geometry, row.geometry):
                results.append(self._row_to_region(row))
        return results

    def touches(self, geo_id: str) -> list[SpatialRegion]:
        """Return regions that share a boundary with ``geo_id``."""

        return self.neighbors(geo_id, predicate="touches")

    def intersects(self, geo_id: str) -> list[SpatialRegion]:
        """Return regions whose geometries intersect ``geo_id``."""

        return self.neighbors(geo_id, predicate="intersects")

    def within(self, geo_id: str) -> list[SpatialRegion]:
        """Return containing regions for ``geo_id``."""

        return self.neighbors(geo_id, predicate="within")

    def topological_relations(
        self,
        predicates: str | Iterable[str] = ("touches", "within"),
        *,
        include_inverse: bool = False,
        confidence: float = 0.85,
    ) -> list[SpatialRelation]:
        """Compute topological relations for the requested predicates."""

        predicate_list = [predicates] if isinstance(predicates, str) else list(predicates)
        relations: list[SpatialRelation] = []
        for predicate in predicate_list:
            normalized = self._normalize_predicate(predicate)
            if normalized in _SYMMETRIC_PREDICATES:
                relations.extend(
                    self._symmetric_relations(normalized, include_inverse=include_inverse, confidence=confidence)
                )
            else:
                relations.extend(self._directed_relations(normalized, confidence=confidence))
        return relations

    def as_kg_nodes(self, *, confidence: float = 0.95, node_prefix: str = "geo:") -> list[KGNode]:
        """Return KG nodes for all spatial regions."""

        return [region.to_kg_node(confidence=confidence, node_prefix=node_prefix) for region in self.regions()]

    def as_kg_edges(
        self,
        predicates: str | Iterable[str] = ("touches", "within"),
        *,
        include_inverse: bool = False,
        confidence: float = 0.85,
        node_prefix: str = "geo:",
    ) -> list[KGEdge]:
        """Return KG edges for requested topological relations."""

        return [
            relation.to_kg_edge(node_prefix=node_prefix)
            for relation in self.topological_relations(
                predicates,
                include_inverse=include_inverse,
                confidence=confidence,
            )
        ]

    def ingest_into_kg(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        predicates: str | Iterable[str] = ("touches", "within"),
        node_confidence: float = 0.95,
        edge_confidence: float = 0.85,
        include_inverse: bool = False,
        node_prefix: str = "geo:",
    ) -> dict[str, int]:
        """Insert spatial region nodes and topology edges into a KnowledgeGraph."""

        nodes = self.as_kg_nodes(confidence=node_confidence, node_prefix=node_prefix)
        for node in nodes:
            knowledge_graph.add_node(node)
        edges = self.as_kg_edges(
            predicates,
            include_inverse=include_inverse,
            confidence=edge_confidence,
            node_prefix=node_prefix,
        )
        for edge in edges:
            knowledge_graph.add_edge(edge)
        return {"regions": len(nodes), "nodes": len(nodes), "edges": len(edges)}

    def spatial_metadata(self) -> dict[str, Any]:
        """Return compact metadata that can be stored in a Freeman domain schema."""

        return {
            "adapter": "freeman.realworld.spatial_adapter.SpatialAdapter",
            "source": self.source,
            "crs": self.crs,
            "region_count": len(self.region_ids),
            "region_ids": self.region_ids,
            "node_type": SPATIAL_REGION_NODE_TYPE,
            "predicates": sorted(GEO_PREDICATES),
        }

    def annotate_schema(self, schema: Mapping[str, Any], *, key: str = "spatial") -> dict[str, Any]:
        """Return a copy of a domain schema with adapter metadata attached."""

        updated = deep_copy_jsonable(dict(schema))
        metadata = dict(updated.get("metadata", {}))
        metadata[key] = self.spatial_metadata()
        updated["metadata"] = metadata
        return updated

    def _resolve_geo_id_column(self, geo_id_column: str | None) -> str | None:
        if geo_id_column is not None:
            if geo_id_column not in self.gdf.columns:
                raise KeyError(f"geo_id_column not found: {geo_id_column}")
            return geo_id_column
        for candidate in _DEFAULT_GEO_ID_COLUMNS:
            if candidate in self.gdf.columns and self.gdf[candidate].notna().all():
                values = [str(value) for value in self.gdf[candidate].tolist()]
                if len(values) == len(set(values)):
                    return candidate
        return None

    def _resolve_label_column(self, label_column: str | None) -> str | None:
        if label_column is not None:
            if label_column not in self.gdf.columns:
                raise KeyError(f"label_column not found: {label_column}")
            return label_column
        for candidate in _DEFAULT_LABEL_COLUMNS:
            if candidate in self.gdf.columns:
                return candidate
        return self.geo_id_column

    def _prepare_geo_ids(self) -> None:
        geo_ids: list[str] = []
        geometry_keys: list[str] = []
        for offset, (_, row) in enumerate(self.gdf.iterrows()):
            fallback = f"region:{offset}"
            value = row[self.geo_id_column] if self.geo_id_column is not None else None
            geo_id = _clean_geo_id(value, fallback)
            geo_ids.append(geo_id)
            geometry_keys.append(f"{self.geometry_key_prefix}:{geo_id}")
        duplicates = sorted({geo_id for geo_id in geo_ids if geo_ids.count(geo_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate geo ids: {', '.join(duplicates)}")
        self.gdf[_INTERNAL_GEO_ID] = geo_ids
        self.gdf[_INTERNAL_GEOMETRY_KEY] = geometry_keys

    def _row_for_geo_id(self, geo_id: str) -> Any:
        matches = self.gdf[self.gdf[_INTERNAL_GEO_ID] == str(geo_id)]
        if matches.empty:
            raise KeyError(f"Unknown geo_id: {geo_id}")
        return matches.iloc[0]

    def _row_to_region(self, row: Any) -> SpatialRegion:
        geometry = row.geometry
        geo_id = str(row[_INTERNAL_GEO_ID])
        label = _clean_geo_id(row[self.label_column] if self.label_column is not None else None, geo_id)
        properties = _json_properties(row, list(self.gdf.columns))
        return SpatialRegion(
            geo_id=geo_id,
            label=label,
            geometry_key=str(row[_INTERNAL_GEOMETRY_KEY]),
            bounds=_as_float_tuple(geometry.bounds),  # type: ignore[arg-type]
            centroid=_geometry_centroid(geometry),
            area=_geometry_area(geometry),
            properties=properties,
            crs=self.crs,
        )

    def _normalize_predicate(self, predicate: str) -> str:
        raw = str(predicate)
        if raw not in _PREDICATE_TO_RELATION:
            raise ValueError(f"Unsupported spatial predicate: {predicate}")
        relation = _PREDICATE_TO_RELATION[raw]
        if relation == GEO_BORDERED_BY:
            return "touches"
        if relation == GEO_WITHIN_REGION:
            return "within"
        if relation == GEO_INTERSECTS:
            return "intersects"
        raise ValueError(f"Unsupported spatial predicate: {predicate}")

    def _relation_type(self, predicate: str) -> str:
        return {
            "touches": GEO_BORDERED_BY,
            "within": GEO_WITHIN_REGION,
            "intersects": GEO_INTERSECTS,
        }[predicate]

    def _predicate_matches(self, predicate: str, source_geometry: Any, target_geometry: Any) -> bool:
        if predicate == "touches":
            return bool(source_geometry.touches(target_geometry))
        if predicate == "within":
            return bool(source_geometry.within(target_geometry))
        if predicate == "intersects":
            return bool(source_geometry.intersects(target_geometry))
        raise ValueError(f"Unsupported spatial predicate: {predicate}")

    def _symmetric_relations(
        self,
        predicate: str,
        *,
        include_inverse: bool,
        confidence: float,
    ) -> list[SpatialRelation]:
        relations: list[SpatialRelation] = []
        rows = [row for _, row in self.gdf.iterrows()]
        for i, source_row in enumerate(rows):
            for target_row in rows[i + 1 :]:
                if not self._predicate_matches(predicate, source_row.geometry, target_row.geometry):
                    continue
                relation = self._build_relation(source_row, target_row, predicate, confidence=confidence)
                relations.append(relation)
                if include_inverse:
                    relations.append(self._build_relation(target_row, source_row, predicate, confidence=confidence))
        return relations

    def _directed_relations(self, predicate: str, *, confidence: float) -> list[SpatialRelation]:
        relations: list[SpatialRelation] = []
        rows = [row for _, row in self.gdf.iterrows()]
        for source_row in rows:
            for target_row in rows:
                if str(source_row[_INTERNAL_GEO_ID]) == str(target_row[_INTERNAL_GEO_ID]):
                    continue
                if self._predicate_matches(predicate, source_row.geometry, target_row.geometry):
                    relations.append(self._build_relation(source_row, target_row, predicate, confidence=confidence))
        return relations

    def _build_relation(self, source_row: Any, target_row: Any, predicate: str, *, confidence: float) -> SpatialRelation:
        source_geometry = source_row.geometry
        target_geometry = target_row.geometry
        source_area = _geometry_area(source_geometry)
        target_area = _geometry_area(target_geometry)
        intersection_area = _geometry_area(source_geometry.intersection(target_geometry))
        metadata = {
            "source_geometry_key": str(source_row[_INTERNAL_GEOMETRY_KEY]),
            "target_geometry_key": str(target_row[_INTERNAL_GEOMETRY_KEY]),
            "source_area": source_area,
            "target_area": target_area,
            "intersection_area": intersection_area,
            "coverage_ratio": intersection_area / source_area if source_area > 0 else 0.0,
            "symmetric": predicate in _SYMMETRIC_PREDICATES,
            "crs": self.crs,
        }
        return SpatialRelation(
            source_geo_id=str(source_row[_INTERNAL_GEO_ID]),
            target_geo_id=str(target_row[_INTERNAL_GEO_ID]),
            predicate=predicate,
            relation_type=self._relation_type(predicate),
            confidence=confidence,
            weight=1.0,
            metadata=metadata,
        )


GeoAdapter = SpatialAdapter

__all__ = [
    "GeoAdapter",
    "GeoDependencyError",
    "SpatialAdapter",
    "SpatialRegion",
    "SpatialRelation",
]
