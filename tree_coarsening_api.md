# Tree Coarsening API Contract

## 1. Scope

A **tree coarsener** replaces a larger labeled directed tree by a smaller labeled directed tree. The design is BPE-like: fitting learns ordered contractions and a vocabulary of learned supertokens; transforming applies those contractions to new trees; decoding expands supertokens back into finer tree structure.

The public data object is always a NetworkX `DiGraph`. Encoders, decoders, vocabularies, provenance tables, and rule metadata are auxiliary artifacts produced and consumed by the package.

The two primitive learned contractions are:

1. **Edge contraction**: contract one encoded parent occurrence together with one encoded child occurrence.
2. **Sibling contraction**: contract several encoded vertices that have the same encoded parent occurrence.

A direct **component contraction** may also materialize an explicitly selected connected raw subtree as one vocabulary recipe. It is a convenience operation equivalent in expressive power to a deterministic sequence of edge contractions; it is used by `NamedVertexCoarsener` and is not statistically learned.

For sibling contraction, “sibling” means sibling in the current encoded tree. Two vertices may share an encoded parent while attaching to different internal sites of that parent after full expansion. This is allowed because encoded edges carry tuple-valued gluing metadata.

---

## 2. Raw input graphs

`fit` receives a sequence of directed rooted trees:

```python
Sequence[nx.DiGraph]
```

Each graph must satisfy:

- the underlying undirected graph is connected;
- `|E| = |V| - 1`;
- exactly one vertex has in-degree `0`;
- every non-root vertex has in-degree `1`;
- all edges are directed from parent to child.

Every raw node must have:

```python
G.nodes[v]["label"]  # str
G.nodes[v]["time"]   # int | float
```

For exact provenance, every raw node should also have:

```python
G.nodes[v]["uid"]
```

If `uid` is missing, the encoder may use the NetworkX node key as the UID, but this fallback must be explicit and validated.

---

## 3. Encoded graph schema

An encoded graph is also a directed rooted tree. Its nodes represent occurrences of vocabulary tokens.

Encoded node keys are opaque NetworkX identifiers. Implementations should usually relabel encoded nodes to simple consecutive integers after transform. The node key is not the token label and is not provenance.

Encoded nodes have exactly the following required attributes:

```python
H.nodes[z]["label"]       # token id, e.g. ("base", "P") or ("star", "P", "S", 4)
H.nodes[z]["super_uids"]  # flat tuple of original UIDs in canonical site order
```

Encoded nodes should not need separate `type`, `uid`, `raw_label`, `time`, `time_span`, or `super_time` attributes. If those quantities are needed, they should be derived from:

1. the token stored in `label`;
2. the occurrence-level `super_uids`; and
3. the graph-level provenance table described below.

Encoded edges carry tuple-valued gluing data:

```python
H.edges[u, v]["attach_map"] = tuple[int, ...]
```

For an encoded edge `u -> v`, if the child token is `T_v`, then:

```python
len(H.edges[u, v]["attach_map"]) == root_count(T_v)
```

The `k`-th entry says which expanded site of the parent token receives the `k`-th exposed root of the child token.

A scalar `attach_index` may be accepted at user boundaries as shorthand for the one-root case, but the internal schema should normalize to tuple-valued `attach_map`.

---

## 4. Token ids

A **token id** is any hashable object, but tuple tokens are recommended because they are explicit and easy to inspect.

Base tokens use:

```python
("base", raw_label)
```

For example:

```python
("base", "P")
```

`EdgeBPECoarsener` uses:

```python
("edge_bpe", rank)
```

where `rank` is the zero-based order in which the edge merge was learned. The expanded structure is recorded in the vocabulary entry, not repeated in the token id.

`StarCoarsener` uses:

```python
("star", parent_label, child_label, arity)
```

For example:

```python
("star", "P", "S", 4)
```

means that four children with raw label `"S"` were contracted under an encoded parent whose raw label was `"P"` when the rule was learned/applied.

For `EdgeBPECoarsener(num_merges=None, min_pair_count=2)`, fitting repeatedly selects the most frequent encoded edge `(parent_token, child_token, attach_map)` with count at least `min_pair_count` and creates one staged edge token. The score is the raw number of matching live edges, including overlapping occurrences. The rule application then contracts a deterministic vertex-disjoint subset of those occurrences. `history_[i]["count"]` stores the raw score and `history_[i]["actual_events"]` stores the number contracted. `num_merges` caps the number of learned rules; `None` means continue until no eligible edge remains.

Because this coarsener uses edge contractions only, every token it creates has exactly one exposed root. Its private compact representation therefore stores an incoming attachment as one integer site. Vocabulary entries and encoded NetworkX edges still use the general tuple-valued `A` and `attach_map` schema.

For `StarCoarsener(d, m, contract_d=None)`, `d` is the witness threshold and `m` is the minimum number of witnesses needed to learn a pair. After a pair is learned, `contract_d` is the transform-time threshold; it defaults to `d` and must satisfy `2 <= contract_d <= d`. The fitted vocabulary is closed after `fit`, so only arities observed during fitting for learned pairs are contracted during transform.

`NamedVertexCoarsener` selects raw vertices either by UID or by raw string label:

```python
NamedVertexCoarsener(uids={"u17", "u18"}, component_policy="all")
NamedVertexCoarsener(labels={"A", "B"}, component_policy="largest")
```

The selected vertices induce maximal connected components in the raw tree. `component_policy="all"` contracts every component of size at least two; `"largest"` contracts one largest component, with deterministic tie-breaking. Singletons are left as base tokens. `fit` learns no statistics and creates an empty shared encoder/decoder vocabulary. During `transform`, each previously unseen canonical component recipe is registered lazily under a compact token of the form:

```python
("named_component", selector_kind, component_size, digest)
```

The digest is only a compact token identifier; the complete `(P, L, A)` recipe remains authoritative in the vocabulary. Because registration occurs during `transform`, the encoder and decoder share the same mutable `Vocabulary` object.

---

## 5. Vocabulary entries: staged canonical recipes

The authoritative vocabulary representation is a staged recipe:

```python
P: tuple[int, ...]
L: tuple[token_id, ...]
A: tuple[int, ...]
```

`P` and `L` are position-aligned. `A` is a flat homogeneous vector of integer attachment sites.

Let `n = len(P) = len(L)`. Each index `i` is a recipe position.

- `L[i]` is a component token id.
- `P[i] = j >= 0` means recipe position `j` is the parent component of recipe position `i`.
- `P[i] = -1` means recipe position `i` is externally attached to the parent of the whole token occurrence.
- `A` stores internal attachment maps for all positions with `P[i] >= 0`, in recipe order.

For a position `i` with `P[i] >= 0`, its attachment slice is:

```python
def attachment_slice(entry, i):
    start = sum(root_count(entry.L[h]) for h in range(i) if entry.P[h] >= 0)
    stop = start + root_count(entry.L[i])
    return entry.A[start:stop]
```

For a position with `P[i] == -1`, there is no internal attachment slice.

For a learned token `T = (P, L, A)`:

```python
site_count(T) = sum(site_count(L[i]) for i in range(len(L)))
root_count(T) = sum(root_count(L[i]) for i in range(len(L)) if P[i] == -1)
```

The expanded site order is deterministic: concatenate the expanded site orders of recipe positions in recipe order. The expanded root order is deterministic: concatenate expanded roots of recipe positions with `P[i] == -1`, again in recipe order.

### Why `A` is part of the vocabulary

Staged recipes must distinguish tokens that have the same component structure but different internal gluing.

Suppose `T_AB` represents:

```text
A
└── B
```

with:

```python
P = (-1, 0)
L = (("base", "A"), ("base", "B"))
A = (0,)
```

A later edge token using `T_AB` and `C` can attach `C` to site `0` of `T_AB`:

```python
P = (-1, 0)
L = (T_AB, ("base", "C"))
A = (0,)
```

which represents:

```text
A
├── B
└── C
```

or it can attach `C` to site `1` of `T_AB`:

```python
P = (-1, 0)
L = (T_AB, ("base", "C"))
A = (1,)
```

which represents:

```text
A
└── B
    └── C
```

These are different vocabulary entries because they decode to different trees.

---

## 6. Provenance and reconstruction

`super_uids` records which original vertices are represented by a token occurrence. It is flat, not nested:

```python
len(H.nodes[z]["super_uids"]) == site_count(H.nodes[z]["label"])
```

The order must agree with the token’s canonical expanded site order.

For a base token occurrence:

```python
H.nodes[z]["label"] = ("base", "P")
H.nodes[z]["super_uids"] = ("g0_n7",)
```

For a star token occurrence:

```python
H.nodes[z]["label"] = ("star", "P", "S", 4)
H.nodes[z]["super_uids"] = ("g0_s10", "g0_s11", "g0_s12", "g0_s13")
```

Exact reconstruction of raw attributes uses a graph-level provenance table:

```python
H.graph["tree_coarsening_provenance"] = {
    "uid_attr": "uid",
    "node_attrs_by_uid": {
        uid: original_node_attribute_dict,
        ...,
    },
}
```

This keeps encoded node attributes small. Times, raw labels, original node attributes, and original UIDs are derived from `super_uids` and the provenance table.

---

## 7. Constructing new staged vocabulary entries

### Edge contraction

Suppose the encoded graph contains an edge:

```text
u -> v
```

with:

```python
M = H.edges[u, v]["attach_map"]
T_u = H.nodes[u]["label"]
T_v = H.nodes[v]["label"]
```

The staged recipe for the new edge token is:

```python
P = (-1, 0)
L = (T_u, T_v)
A = M
```

This naturally creates different vocabulary entries when the same child token attaches to different internal sites of the same parent token.

### Sibling contraction

Suppose `v_1, ..., v_r` are contracted as siblings under the same encoded parent `u`. Let:

```python
T_i = H.nodes[v_i]["label"]
M_i = H.edges[u, v_i]["attach_map"]
```

The staged recipe for the new sibling token is:

```python
P = (-1, -1, ..., -1)
L = (T_1, T_2, ..., T_r)
A = ()
```

The replacement edge from `u` to the new token stores the concatenated attachment maps:

```python
attach_map = M_1 + M_2 + ... + M_r
```

This construction only requires the selected vertices to share the same encoded parent. It does not require their incoming `attach_map`s to be equal.

---

## 8. Encoder, decoder, and coarsener API

### Vocabulary dataclass

```python
@dataclass(frozen=True)
class VocabEntry:
    token: Hashable
    parent: tuple[int, ...]
    label: tuple[Hashable, ...]
    attach: tuple[int, ...]
    created_at_step: int
    operation: Literal["base", "edge", "siblings", "component"]
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### Encoder

```python
class TreeEncoder:
    model_id: str
    vocab: Vocabulary
    rules: Sequence[EncodingRule]
    base_labels: set[str]

    label_attr: str = "label"
    time_attr: str = "time"
    uid_attr: str = "uid"
    super_uid_attr: str = "super_uids"
    attach_attr: str = "attach_map"

    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph: ...
```

Encoding initializes raw vertices as base tokens, initializes raw edges with `attach_map=(0,)`, and then applies learned rules in temporal order.

### Decoder

```python
class TreeDecoder:
    model_id: str
    vocab: Vocabulary
    base_labels: set[str]

    label_attr: str = "label"
    time_attr: str = "time"
    uid_attr: str = "uid"
    super_uid_attr: str = "super_uids"
    attach_attr: str = "attach_map"

    def decode(
        self,
        H: nx.DiGraph,
        *,
        target: Any | None = None,
        by: Literal["node", "label"] = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph: ...
```

Recommended behavior:

- `target is None`: decode the whole graph;
- `by="node"`: decode one encoded node occurrence;
- `by="label"`: decode every occurrence with the given token label;
- `recursive=False`: expand one recipe layer where possible;
- `recursive=True`: continue expanding selected tokens until base labels are reached.

### Abstract coarsener

```python
class TreeCoarsener(ABC):
    encoder_: TreeEncoder | None = None
    decoder_: TreeDecoder | None = None

    def fit(self, graphs: Sequence[nx.DiGraph]) -> Self: ...
    def transform(self, graph: nx.DiGraph) -> nx.DiGraph: ...
    def inverse_transform(self, graph: nx.DiGraph, **decode_kwargs) -> nx.DiGraph: ...
```

This supports:

```python
H = coarsener.fit(X).transform(Y)
encoder = coarsener.encoder_
decoder = coarsener.decoder_
```

---

## 9. Partial decoding and boundary policy

Full recursive decoding is always unambiguous.

Partial decoding can expose a boundary problem. If expanding a parent token would require one still-collapsed child token to have multiple encoded parents, the result would not be a directed tree. Therefore:

- `boundary_policy="expand"` minimally expands the boundary child enough to preserve a tree;
- `boundary_policy="raise"` raises a validation error.

This is the cost of allowing broad sibling contractions under a supernode parent. They are valid and exactly decodable, but some partial decodes need boundary-aware expansion.

---

## 10. Composition

Sequentially fitted encoders should compose lazily first:

```python
def combine(
    encoders: Sequence[TreeEncoder],
    decoders: Sequence[TreeDecoder],
    *,
    mode: Literal["lazy", "materialized"] = "lazy",
    validate: bool = True,
) -> tuple[TreeEncoder, TreeDecoder]: ...
```

The encoders and decoders are supplied in encoder application order:

```python
combined_encoder, combined_decoder = combine(
    encoders=[encoder1, encoder2, encoder3],
    decoders=[decoder1, decoder2, decoder3],
)
```

Then:

```python
combined_encoder.encode(G)
```

means:

```python
encoder3.encode(encoder2.encode(encoder1.encode(G)))
```

and:

```python
combined_decoder.decode(H)
```

means:

```python
decoder1.decode(decoder2.decode(decoder3.decode(H)))
```

Materialized composition is useful but requires careful root/site coordinate translation and should come after the lazy version has tests.

---

## 11. Locked design decisions

1. Vocabulary entries are staged recipes `(P, L, A)`.
2. `A` is a flat integer attachment vector.
3. Encoded node `label` is the token id; there is no separate required `type` field.
4. Encoded node keys are opaque and should usually be simple consecutive integers.
5. Encoded nodes store flat `super_uids`, not nested `super_label`.
6. Raw node attributes, including time, are recovered from graph-level provenance.
7. Encoded edges use tuple-valued `attach_map`.
8. Sibling means sibling in the encoded tree.
9. Broad sibling contractions are valid and exactly decodable.
10. Partial decoding is boundary-aware.
11. Composition should start lazy.
