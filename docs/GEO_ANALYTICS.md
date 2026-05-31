# Freeman Geo Analytics

This document defines the lightweight regional-analytics layer used by Freeman.
The goal is to make spatial structure visible to the simulator and causal graph
without requiring PostGIS or a heavy geospatial database.

## Formal Model

Let:

- \(A\) be the set of domain actors.
- \(R\) be the set of spatial regions.
- \(g: A \rightarrow R\) map each actor to a region.
- \(W \in \mathbb{R}^{|R|\times |R|}\) be a weighted spatial adjacency matrix.

For every pair of actors \(a_i, a_j \in A\), Freeman materializes a spatial
relation when:

\[
W_{g(a_i), g(a_j)} > 0,\quad a_i \ne a_j.
\]

The generated relation is:

```json
{
  "source_id": "actor_i",
  "target_id": "actor_j",
  "relation_type": "spatial_neighbor",
  "weights": {
    "adjacency": 1.0,
    "spatial_weight": 1.0
  }
}
```

By default adjacency is treated as undirected, so Freeman emits both
\(a_i \rightarrow a_j\) and \(a_j \rightarrow a_i\). Set `spatial.directed=true`
or edge-level `directed=true` for one-way spatial influence.

## Schema Contract

Attach spatial metadata at the top level of a domain schema:

```json
{
  "domain_id": "regional_water_risk",
  "actors": [
    {
      "id": "country_a",
      "name": "Country A",
      "state": {"influence": 0.6},
      "metadata": {"geo_id": "REGION_A"}
    },
    {
      "id": "country_b",
      "name": "Country B",
      "state": {"influence": 0.5},
      "metadata": {"geo_id": "REGION_B"}
    }
  ],
  "spatial": {
    "adjacency": [
      {"source": "REGION_A", "target": "REGION_B", "weight": 0.7}
    ],
    "relation_type": "spatial_neighbor"
  }
}
```

Actor-region mapping can also be explicit:

```json
{
  "spatial": {
    "actor_regions": {
      "country_a": "REGION_A",
      "country_b": "REGION_B"
    },
    "adjacency": {
      "REGION_A": {"REGION_B": {"weight": 0.7, "distance_km": 240.0}}
    }
  }
}
```

Supported adjacency forms:

- `{"REGION_A": ["REGION_B", "REGION_C"]}`
- `{"REGION_A": {"REGION_B": {"weight": 0.7}}}`
- `[["REGION_A", "REGION_B", 0.7]]`
- `[{"source": "REGION_A", "target": "REGION_B", "weight": 0.7}]`

Supported actor metadata keys:

- `geo_id`
- `region_id`
- `spatial_region`
- `spatial_region_id`
- `iso_a3`
- nested `metadata.geo.geo_id`
- nested `metadata.spatial.region_id`

## Compile-Time Materialization

During `DomainCompiler.compile(schema)`, Freeman calls:

```python
initialize_spatial_relations(world)
```

before returning the compiled `WorldState`. The real-world manifold path also
uses the same function idempotently before simulation. The function reads
`world.metadata.get("spatial")`, adds any
missing `Relation` objects to `world.relations`, and writes a compact trace to:

```json
{
  "_spatial_materialization": {
    "relation_count": 2,
    "relations": [
      {
        "source_actor": "country_a",
        "target_actor": "country_b",
        "source_region": "REGION_A",
        "target_region": "REGION_B",
        "relation_type": "spatial_neighbor",
        "weight": 0.7
      }
    ]
  }
}
```

The operation is idempotent: repeated calls do not duplicate existing relations
with the same `(source_id, target_id, relation_type)`.

## Large Graph Diagnostics

Spatial materialization is sparse-adjacency driven, but actor-pair expansion can
still become expensive when many actors share adjacent regions. Freeman logs a
warning through `freeman.domain.spatial` when either:

- mapped region count exceeds `SPATIAL_REGION_WARNING_THRESHOLD = 100`
- estimated materialized relations exceed `SPATIAL_RELATION_WARNING_THRESHOLD = 10000`

The warning includes `domain_id`, mapped region count, adjacency-edge count and
estimated relation count. It does not stop compilation; it is a signal to check
whether the schema should aggregate actors, prune weak adjacency edges or move to
a dedicated sparse spatial operator.

## Vector Geometry Adapter

`freeman.realworld.spatial_adapter.SpatialAdapter` handles vector geometry:

- load GeoJSON/Shapefile into a GeoDataFrame
- compute `touches`, `within`, `intersects`
- export `SpatialRegion` nodes to KG
- export topology edges such as `geo:borderedBy`, `geo:withinRegion`, `geo:intersects`

Install optional dependencies with:

```bash
pip install ".[geo]"
```

Minimal usage:

```python
from freeman.memory.knowledgegraph import KnowledgeGraph
from freeman.realworld.spatial_adapter import SpatialAdapter

adapter = SpatialAdapter.from_file("regions.geojson", geo_id_column="iso_a3", label_column="name")
kg = KnowledgeGraph(auto_load=False, auto_save=False)
adapter.ingest_into_kg(kg, predicates=("touches", "within"))
```

## Current Scope and Limits

This layer currently materializes spatial structure into actor relations. It does
not automatically create numeric resource diffusion or causal DAG edges. Use it
when the model needs geography-aware graph context, retrieval, or causal
reasoning over neighboring regions.

For numeric spatial dynamics, define explicit `actor_update_rules` or resource
`coupling_weights` that consume actor/resource state. A future extension can map
`spatial.adjacency` into resource-level diffusion terms:

\[
x_i(t+1) = \alpha x_i(t) + \lambda \sum_j W_{ij}(x_j(t) - x_i(t)).
\]
