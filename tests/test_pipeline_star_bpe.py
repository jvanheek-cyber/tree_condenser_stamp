from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import (
    CompositeType,
    EdgeBPECoarsener,
    StarCoarsener,
    ValidationError,
    combine,
)
from tree_coarsening.utils import make_starburst_dataset


def uid_edges(graph: nx.DiGraph) -> set[tuple[object, object]]:
    return {
        (graph.nodes[u].get("uid", u), graph.nodes[v].get("uid", v))
        for u, v in graph.edges
    }


def encoded_occurrences(graph: nx.DiGraph):
    return sorted(
        (
            tuple(data["super_uids"]),
            repr(data["label"]),
            repr(data["type"]),
            data["size"],
            data["time"],
        )
        for _, data in graph.nodes(data=True)
    )


def encoded_edges_by_uid_groups(graph: nx.DiGraph):
    nodes = {node: tuple(data["super_uids"]) for node, data in graph.nodes(data=True)}
    return sorted(
        (nodes[u], nodes[v], tuple(data["attach_map"]))
        for u, v, data in graph.edges(data=True)
    )


def test_star_then_bpe_fit_transform_and_reverse_decode() -> None:
    raw = make_starburst_dataset(
        n_graphs=5,
        seed=21,
        max_nodes=18,
        n_bursts=3,
        burst_size_range=(3, 5),
        tail_probability=0.5,
        parent_label="P",
        child_label="S",
        tail_label="T",
    )
    star = StarCoarsener(d=3, m=1, contract_d=3).fit(raw)
    star_graphs = star.transform(raw)

    bpe = EdgeBPECoarsener(num_merges=12, min_pair_count=1).fit(star_graphs)
    final_graphs = bpe.transform(star_graphs)

    for original, star_graph, final_graph in zip(raw, star_graphs, final_graphs):
        assert all(
            isinstance(data["label"], (str, tuple))
            for _, data in star_graph.nodes(data=True)
        )
        assert any(
            isinstance(data["type"], CompositeType)
            for _, data in final_graph.nodes(data=True)
        )

        recovered_star = bpe.inverse_transform(final_graph)
        assert encoded_occurrences(recovered_star) == encoded_occurrences(star_graph)
        assert encoded_edges_by_uid_groups(recovered_star) == encoded_edges_by_uid_groups(
            star_graph
        )

        recovered_raw = star.inverse_transform(recovered_star)
        assert uid_edges(recovered_raw) == uid_edges(original)
        assert {
            data["uid"]: (data["label"], data["time"])
            for _, data in recovered_raw.nodes(data=True)
        } == {
            data["uid"]: (data["label"], data["time"])
            for _, data in original.nodes(data=True)
        }


def test_lazy_combination_matches_explicit_pipeline() -> None:
    raw = make_starburst_dataset(
        n_graphs=3,
        seed=31,
        max_nodes=14,
        n_bursts=2,
        burst_size_range=(3, 4),
        parent_label="P",
        child_label="S",
    )
    star = StarCoarsener(d=3, m=1).fit(raw)
    intermediate = star.transform(raw)
    bpe = EdgeBPECoarsener(num_merges=8, min_pair_count=1).fit(intermediate)

    encoder, decoder = combine(
        [star.encoder_, bpe.encoder_],
        [star.decoder_, bpe.decoder_],
    )
    explicit = bpe.transform(star.transform(raw[0]))
    combined = encoder.encode(raw[0])
    assert encoded_occurrences(combined) == encoded_occurrences(explicit)
    assert uid_edges(decoder.decode(combined)) == uid_edges(raw[0])


def test_bpe_then_star_uses_the_same_graph_contract() -> None:
    def repeated_branches(prefix: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node(0, label="R", time=0.0, uid=f"{prefix}r")
        next_node = 1
        for branch in range(4):
            parent, child = next_node, next_node + 1
            next_node += 2
            graph.add_node(
                parent,
                label="A",
                time=1.0 + branch / 10,
                uid=f"{prefix}a{branch}",
            )
            graph.add_node(
                child,
                label="B",
                time=2.0 + branch / 10,
                uid=f"{prefix}b{branch}",
            )
            graph.add_edge(0, parent)
            graph.add_edge(parent, child)
        return graph

    raw = [repeated_branches("x"), repeated_branches("y")]
    bpe = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(raw)
    intermediate = bpe.transform(raw)
    star = StarCoarsener(d=3, m=1, contract_d=3).fit(intermediate)
    final = star.transform(intermediate)

    for original, encoded in zip(raw, final):
        recovered_bpe = star.inverse_transform(encoded)
        recovered_raw = bpe.inverse_transform(recovered_bpe)
        assert uid_edges(recovered_raw) == uid_edges(original)


def test_fit_rejects_mixed_raw_and_transformed_batches() -> None:
    raw = make_starburst_dataset(
        n_graphs=2,
        seed=44,
        max_nodes=10,
        n_bursts=2,
        burst_size_range=(3, 4),
        parent_label="P",
        child_label="S",
    )
    star = StarCoarsener(d=3, m=1).fit(raw)
    transformed = star.transform(raw[0])

    with pytest.raises(ValidationError, match="cannot mix raw trees"):
        EdgeBPECoarsener(num_merges=2, min_pair_count=1).fit(
            [raw[1], transformed]
        )


def _attachment_variant_tree(site: int, *, suffix: str) -> nx.DiGraph:
    from tree_coarsening import base_token
    from tree_coarsening.provenance import NODE_ATTRS_KEY, PROVENANCE_KEY
    from tree_coarsening.schema import RAW_INPUT_FLAG

    parent_type = CompositeType(
        model_id="previous-stage",
        kind="component",
        label="P2",
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
        label="P2",
        type=parent_type,
        size=2,
        time=1.0,
        uid=f"parent-{suffix}",
        super_label=(f"a-{suffix}", f"b-{suffix}"),
        super_uids=(f"a-{suffix}", f"b-{suffix}"),
    )
    graph.add_node(
        1,
        label="C",
        type=base_token("C"),
        size=1,
        time=2.0,
        uid=f"child-{suffix}",
        super_label=f"c-{suffix}",
        super_uids=(f"c-{suffix}",),
    )
    graph.add_edge(0, 1, attach_map=(site,))
    graph.graph[RAW_INPUT_FLAG] = False
    graph.graph[PROVENANCE_KEY] = {
        NODE_ATTRS_KEY: {
            f"a-{suffix}": {"uid": f"a-{suffix}", "label": "A", "time": 0.0},
            f"b-{suffix}": {"uid": f"b-{suffix}", "label": "B", "time": 1.0},
            f"c-{suffix}": {"uid": f"c-{suffix}", "label": "C", "time": 2.0},
        },
        "uid_attr": "uid",
    }
    return graph


def test_bpe_fit_aggregates_attachment_variants_but_transform_preserves_them() -> None:
    graphs = [
        _attachment_variant_tree(0, suffix="left"),
        _attachment_variant_tree(1, suffix="right"),
    ]
    bpe = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(graphs)

    assert bpe.history_[0]["count"] == 2
    assert bpe.history_[0]["parent_label"] == "P2"
    assert bpe.history_[0]["child_label"] == "C"
    assert "attach_map" not in bpe.history_[0]

    encoded = bpe.transform(graphs)
    assert [next(iter(graph.nodes(data=True)))[1]["label"] for graph in encoded] == [
        ("edge_bpe", 0),
        ("edge_bpe", 0),
    ]
    exact_types = [next(iter(graph.nodes(data=True)))[1]["type"] for graph in encoded]
    assert [exact.attach for exact in exact_types] == [(0,), (1,)]

    recovered = bpe.inverse_transform(encoded[1])
    assert tuple(recovered.edges[0, 1]["attach_map"]) == (1,)


def test_named_component_can_follow_star_stage() -> None:
    from tree_coarsening import NamedVertexCoarsener, star_token

    graph = nx.DiGraph()
    graph.add_node(0, label="P", time=0.0, uid="p")
    for node in range(1, 4):
        graph.add_node(node, label="S", time=float(node), uid=f"s{node}")
        graph.add_edge(0, node)

    star = StarCoarsener(d=3, m=1).fit([graph])
    star_graph = star.transform(graph)
    star_label = star_token("P", "S", 3)

    named = NamedVertexCoarsener(labels={"P", star_label}).fit([star_graph])
    contracted = named.transform(star_graph)
    assert contracted.number_of_nodes() == 1
    assert isinstance(next(iter(contracted.nodes(data=True)))[1]["type"], CompositeType)

    recovered_star = named.inverse_transform(contracted)
    assert encoded_occurrences(recovered_star) == encoded_occurrences(star_graph)
    assert encoded_edges_by_uid_groups(recovered_star) == encoded_edges_by_uid_groups(
        star_graph
    )
    recovered_raw = star.inverse_transform(recovered_star)
    assert uid_edges(recovered_raw) == uid_edges(graph)


def test_single_graph_pipeline_matches_requested_interface() -> None:
    raw = make_starburst_dataset(
        n_graphs=1,
        seed=71,
        max_nodes=16,
        n_bursts=3,
        burst_size_range=(3, 5),
        parent_label="P",
        child_label="S",
    )[0]

    star = StarCoarsener(d=3, m=1, contract_d=3)
    star.fit(raw)
    intermediate = star.transform(raw)
    assert isinstance(intermediate, nx.DiGraph)

    bpe = EdgeBPECoarsener(num_merges=10, min_pair_count=1)
    bpe.fit(intermediate)
    final = bpe.transform(intermediate)
    assert isinstance(final, nx.DiGraph)

    recovered_intermediate = bpe.inverse_transform(final)
    recovered_raw = star.inverse_transform(recovered_intermediate)
    assert uid_edges(recovered_raw) == uid_edges(raw)
