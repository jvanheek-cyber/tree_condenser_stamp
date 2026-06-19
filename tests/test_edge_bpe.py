from __future__ import annotations

import networkx as nx

from tree_coarsening import (
    CompositeType,
    EdgeBPECoarsener,
    base_token,
    edge_bpe_token,
)
from tree_coarsening.provenance import NODE_ATTRS_KEY, PROVENANCE_KEY


EXPECTED_ENCODED_NODE_FIELDS = {
    "label", "type", "size", "time", "super_label", "super_uids"
}


def path_graph(labels: list[str], *, prefix: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i, label in enumerate(labels):
        graph.add_node(i, label=label, time=float(i), uid=f"{prefix}{i}")
        if i:
            graph.add_edge(i - 1, i)
    return graph


def uid_edge_set(graph: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (graph.nodes[u].get("uid", u), graph.nodes[v].get("uid", v))
        for u, v in graph.edges
    }


def iter_composite_types(type_token):
    if isinstance(type_token, CompositeType):
        yield type_token
        for child_type in type_token.component_types:
            yield from iter_composite_types(child_type)


def encoded_attachment_variant(site: int, *, prefix: str) -> nx.DiGraph:
    """Same fitting labels/specifications, different occurrence attachment."""

    parent_type = CompositeType(
        model_id="prior-stage",
        kind="component",
        label="P",
        parent=(-1, 0),
        component_labels=("A", "B"),
        component_types=(base_token("A"), base_token("B")),
        component_sizes=(1, 1),
        component_root_counts=(1, 1),
        attach=(0,),
    )
    graph = nx.DiGraph()
    graph.add_node(
        0,
        label="P",
        type=parent_type,
        size=2,
        time=1.0,
        super_label=(f"{prefix}a", f"{prefix}b"),
        super_uids=(f"{prefix}a", f"{prefix}b"),
    )
    graph.add_node(
        1,
        label="C",
        type=base_token("C"),
        size=1,
        time=2.0,
        super_label=f"{prefix}c",
        super_uids=(f"{prefix}c",),
    )
    graph.add_edge(0, 1, attach_map=(site,))
    graph.graph[PROVENANCE_KEY] = {
        NODE_ATTRS_KEY: {
            f"{prefix}a": {"uid": f"{prefix}a", "label": "A", "time": 0.0},
            f"{prefix}b": {"uid": f"{prefix}b", "label": "B", "time": 1.0},
            f"{prefix}c": {"uid": f"{prefix}c", "label": "C", "time": 2.0},
        },
        "uid_attr": "uid",
    }
    return graph


def test_edge_bpe_learns_label_rule_and_roundtrips() -> None:
    trees = [
        path_graph(["A", "B", "C"], prefix="a"),
        path_graph(["A", "B", "D"], prefix="b"),
    ]
    coarsener = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(trees)

    token = edge_bpe_token(0)
    assert token in coarsener.encoder_.vocab.symbols
    assert token not in coarsener.encoder_.vocab.entries
    rule = coarsener.encoder_.edge_rules[0]
    assert (rule.parent_label, rule.child_label) == ("A", "B")
    assert "attach_map" not in coarsener.history_[0]

    encoded = coarsener.transform(trees[0])
    assert list(encoded.nodes) == list(range(encoded.number_of_nodes()))
    assert any(data["label"] == token for _, data in encoded.nodes(data=True))
    assert all(set(data) == EXPECTED_ENCODED_NODE_FIELDS for _, data in encoded.nodes(data=True))

    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(trees[0])


def test_edge_bpe_attachment_is_recorded_only_in_exact_transform_type() -> None:
    # A-B is the unique most frequent initial pair; after that merge,
    # (A-B)-C is the unique most frequent second pair.
    trees = [
        path_graph(["A", "B", "C"], prefix="a"),
        path_graph(["A", "B", "C"], prefix="b"),
        path_graph(["A", "B", "D"], prefix="c"),
    ]
    coarsener = EdgeBPECoarsener(num_merges=2, min_pair_count=1).fit(trees)

    # Fitted rules contain label pairs but no attachment sites.
    assert all("attach_map" not in item for item in coarsener.history_)

    encoded_graphs = coarsener.transform(trees)
    exact_types = [
        exact
        for encoded in encoded_graphs
        for _, data in encoded.nodes(data=True)
        for exact in iter_composite_types(data["type"])
        if exact.kind == "edge_bpe"
    ]
    assert exact_types
    # The nested merge that appends C to the A-B token attaches at its internal
    # B site, demonstrating deterministic occurrence-specific realization.
    assert any(exact.attach == (1,) for exact in exact_types)

    for original, encoded in zip(trees, encoded_graphs):
        decoded = coarsener.decode(encoded)
        assert uid_edge_set(decoded) == uid_edge_set(original)


def test_edge_bpe_counts_attachment_variants_as_one_label_pair() -> None:
    inputs = [
        encoded_attachment_variant(0, prefix="x"),
        encoded_attachment_variant(1, prefix="y"),
    ]
    coarsener = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(inputs)

    assert coarsener.history_[0]["parent_label"] == "P"
    assert coarsener.history_[0]["child_label"] == "C"
    assert coarsener.history_[0]["count"] == 2
    assert "attach_map" not in coarsener.history_[0]

    outputs = coarsener.transform(inputs)
    exact_types = [next(iter(graph.nodes(data=True)))[1]["type"] for graph in outputs]
    assert all(isinstance(exact, CompositeType) for exact in exact_types)
    assert {exact.attach for exact in exact_types} == {(0,), (1,)}
    assert {exact.label for exact in exact_types} == {edge_bpe_token(0)}

    recovered = coarsener.inverse_transform(outputs)
    for original, decoded in zip(inputs, recovered):
        assert uid_edge_set(decoded) == uid_edge_set(original)
        assert next(iter(decoded.edges(data=True)))[2]["attach_map"] == next(
            iter(original.edges(data=True))
        )[2]["attach_map"]


def test_edge_bpe_partial_decode_roundtrips() -> None:
    graph = path_graph(["A", "B", "C"], prefix="x")
    coarsener = EdgeBPECoarsener(num_merges=2, min_pair_count=1).fit([graph])
    encoded = coarsener.transform(graph)
    learned_nodes = [
        node
        for node, data in encoded.nodes(data=True)
        if isinstance(data["type"], CompositeType)
    ]
    assert learned_nodes

    partial = coarsener.decode(
        encoded, target=learned_nodes[0], by="node", recursive=False
    )
    assert all(set(data) == EXPECTED_ENCODED_NODE_FIELDS for _, data in partial.nodes(data=True))
    assert uid_edge_set(coarsener.decode(partial)) == uid_edge_set(graph)


def test_edge_bpe_scores_raw_edges_before_overlap_filtering() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, label="A", time=0.0, uid="root")
    for i in range(1, 9):
        graph.add_node(i, label="B", time=float(i), uid=f"b{i}")
        graph.add_edge(0, i)

    coarsener = EdgeBPECoarsener(num_merges=1, min_pair_count=1).fit([graph])
    assert coarsener.history_[0]["count"] == 8
    assert coarsener.history_[0]["count_semantics"] == "raw_matching_edges"
    assert coarsener.history_[0]["actual_events"] == 1


def test_edge_bpe_long_repeated_merge_preserves_uid_order() -> None:
    graph = path_graph(["A"] * 256, prefix="p")
    coarsener = EdgeBPECoarsener(num_merges=20, min_pair_count=1).fit([graph])
    encoded = coarsener.transform(graph)

    flattened = tuple(
        uid for _, data in encoded.nodes(data=True) for uid in data["super_uids"]
    )
    assert set(flattened) == {f"p{i}" for i in range(256)}
    assert len(flattened) == 256
    assert uid_edge_set(coarsener.decode(encoded)) == uid_edge_set(graph)
