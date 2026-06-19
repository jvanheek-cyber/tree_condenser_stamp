# tree-coarsening

Experimental BPE-like coarsening for labeled directed trees represented as NetworkX `DiGraph`s.

The core object is always a NetworkX graph. Fitted coarseners produce an encoder, a decoder, and a staged vocabulary of supertokens.

## Common graph contract

`fit`, `transform`, and `inverse_transform` accept either one `nx.DiGraph` or a
nonempty sequence of graphs. Output shape matches input shape.

Raw user nodes still need only:

```python
G.nodes[v]["label"]  # usually str
G.nodes[v]["time"]   # int | float
G.nodes[v]["uid"]    # recommended; falls back to node key
```

The base class normalizes raw and previously transformed trees to the common
fitting contract:

```python
node["label"]  # hashable fitting symbol
node["size"]   # positive represented-site count
node["time"]   # scalar representative time
```

Transformed nodes also carry exact structural/provenance information:

```python
node["type"]         # exact attachment-sensitive decoder type
node["super_label"]  # recipe-aligned provenance payload
node["super_uids"]   # backward-compatible flat UID order
```

Every contraction uses:

```python
new_size = sum(component sizes)
new_time = max(component times)
```

Encoded edges continue to use tuple-valued `attach_map`. Attachment data is for
transform and decode, not the common fitting abstraction.

Version 0.9 retains the completed two-pass pipeline migration and re-enables the
attachment-independent Numba fitter. Every coarsener fits
the same `label + size + time + topology` abstraction. Exact attachment-sensitive
structure lives in `type` and `attach_map`, which are used during transform and
decode but not during BPE pair counting.

A clean staged workflow is now supported directly:

```python
star = StarCoarsener(d=3, m=1).fit(raw_trees)
U = star.transform(raw_trees)

bpe = EdgeBPECoarsener(num_merges=32, min_pair_count=2).fit(U)
V = bpe.transform(U)

U_recovered = bpe.inverse_transform(V)
raw_recovered = star.inverse_transform(U_recovered)
```

## Example

```python
from tree_coarsening import StarCoarsener
from tree_coarsening.utils import make_starburst_dataset

X = make_starburst_dataset(n_graphs=3, seed=0)
coarsener = StarCoarsener(d=3, m=1).fit(X)

H = coarsener.transform(X[0])
G_roundtrip = coarsener.decode(H)
```


### Edge-only BPE example

```python
from tree_coarsening import EdgeBPECoarsener

coarsener = EdgeBPECoarsener(num_merges=32, min_pair_count=2).fit(X)
H = coarsener.transform(X[0])
G_roundtrip = coarsener.decode(H)
```

### Clean Star → BPE pipeline

```python
star = StarCoarsener(d=3, m=1).fit(raw_trees)
U = star.transform(raw_trees)

bpe = EdgeBPECoarsener(num_merges=32, min_pair_count=2).fit(U)
V = bpe.transform(U)

U_recovered = bpe.inverse_transform(V)
raw_recovered = star.inverse_transform(U_recovered)
```

No adapter, previous-stage vocabulary, or user-supplied context is required.

An experimental Numba fitting backend is available as an optional extra:

```bash
pip install -e ".[numba]"
```

```python
coarsener = EdgeBPECoarsener(
    num_merges=32,
    min_pair_count=2,
    backend="numba",
).fit(X)
```

The default remains `backend="python"`. `backend="numba"` now uses the
same attachment-independent `(parent_label, child_label)` rule key and preserves
raw-count scores, additive sizes, maximum component times, deterministic overlap
resolution, and learned-rule history. Only fitting is compiled; transform and
decode remain Python. The first Numba fit includes JIT warm-up, so benchmark
first-call and warmed timings separately.

`EdgeBPECoarsener` learns only edge contractions. At each step it selects the
most frequent fitting-label pair `(parent_label, child_label)` with count at
least `min_pair_count`, creates a token of the form:

```python
("edge_bpe", rank)
```

and contracts a deterministic non-overlapping subset of those occurrences.
The score is the raw matching-edge count, including overlapping matches; each
history record also stores `actual_events`, the number contracted in that pass.
Attachment maps do not affect fitting counts. During transform, each concrete
occurrence deterministically creates an exact `CompositeType` containing its
actual attachment map and component types.

The Python fitting implementation keeps states deliberately lean: it uses
compact arrays for parent pointers, child lists, integer fitting labels, sizes,
times, and liveness; contracts in place; updates pair counts locally; and omits
attachments and UID provenance. During transform, the occurrence's real
`attach_map` is used to construct an exact `CompositeType`, while all variants
of one rule retain the same downstream fitting label.

### Named-vertex component contraction

```python
from tree_coarsening import NamedVertexCoarsener

by_uid = NamedVertexCoarsener(
    uids={"node-17", "node-18", "node-19"},
    component_policy="all",
).fit(X)

by_label = NamedVertexCoarsener(
    labels={"A", "B"},
    component_policy="largest",
).fit(X)

H = by_label.transform(X[0])
G_roundtrip = by_label.decode(H)
```

The selector induces maximal connected components in the current coarsenable
tree. `"all"`
contracts every component of size at least two; `"largest"` contracts one
largest component. UID selection matches represented original UIDs in
`super_uids`; label selection accepts hashable fitting labels. `fit` learns no
statistics. Canonical component recipes are added lazily to the shared
vocabulary during `transform`.

`StarCoarsener(d, m, contract_d=None)` learns label pairs `(P, C)` such that at least `m` vertices with label `P` each have at least `d` children with label `C`. After a pair is learned, `contract_d` controls the smaller transform-time threshold; it defaults to `d`. Transform contracts matching child groups into tokens of the form:

```python
("star", parent_label, child_label, arity)
```

For example, `("star", "P", "S", 4)` means four `"S"` children under a `"P"` parent were contracted.


### Example simulators and notebooks

The non-core simulation helpers include:

```python
from tree_coarsening.utils import (
    make_edge_bpe_dataset,
    make_named_component_tree,
    make_starburst_dataset,
)
```

- `make_edge_bpe_dataset(...)` creates trees with repeated labeled paths, making
  successive edge-only BPE merges easy to inspect.
- `make_named_component_tree(...)` creates separated connected components with
  predictable UID prefixes for label- and UID-based contraction examples.
- `make_starburst_dataset(...)` remains the star-coarsener simulator.

Runnable syntax notebooks are provided in `notebooks/` for all three concrete
coarseners.

## Package layout

```text
tree_coarsening/
  canonical.py
  coarsener.py
  compose.py
  decoder.py
  encoder.py
  exceptions.py
  nx_io.py
  provenance.py
  schema.py
  stage_decoder.py
  structural.py
  coarseners/
    edge_bpe.py
    edge_bpe_numba.py  # optional experimental fit backend
    named_vertices.py
    star.py
  validation.py
  vocabulary.py
  utils/
    simulate.py

docs/
  edge_bpe_numba_experiment.md
  pipeline_migration_pass2.md
  tree_coarsening_api.md
  star_coarsener.md

benchmarks/
  benchmark_edge_bpe_numba.py

notebooks/
  edge_bpe_coarsener_example.ipynb
  named_vertex_coarsener_example.ipynb
  star_coarsener_example.ipynb
```

See `docs/tree_coarsening_api.md` for the API contract and
`docs/pipeline_migration_pass2.md` for the completed migration notes.

## Deterministic round-trip tests

The high-level end-to-end suite lives under:

```text
tests/roundtrip_suite/
```

Run the automated scripts with either:

```bash
pytest -q tests/roundtrip_suite
python tests/roundtrip_suite/run_roundtrip_checks.py
```

The matching executed notebook,
`tests/roundtrip_suite/coarsening_roundtrip_visual_checks.ipynb`, uses the same
seed constants and graph constructors and adds small visual comparisons of raw,
coarsened, intermediate, and decoded trees.
