# tree-coarsening

Experimental BPE-like coarsening for labeled directed trees represented as NetworkX `DiGraph`s.

For developers: please run unit tests:

```python
python -m pip install -e ".[dev,numba]"
python -m pytest
```

before committing. The tests now require that new coarseners add an example tree to factory.py before committing, so that we can easily update API-violation checks on new coarseners.

## Examples

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

