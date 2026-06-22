from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import (
    CompositeType,
    EdgeBPECoarsener,
    EdgeBPEWithAutoStarCoarsener,
    EdgeStarRule,
    edge_star_token,
)


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


def star_graph(parent_label: str, child_label: str, arity: int, *, prefix: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("p", label=parent_label, time=0.0, uid=f"{prefix}p")
    for j in range(arity):
        child = f"{prefix}c{j}"
        graph.add_node(child, label=child_label, time=1.0 + j, uid=child)
        graph.add_edge("p", child)
    return graph


def uid_edge_set(graph: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (graph.nodes[u].get("uid", u), graph.nodes[v].get("uid", v))
        for u, v in graph.edges
    }


def uid_node_set(graph: nx.DiGraph) -> set[str]:
    return {graph.nodes[n].get("uid", n) for n in graph.nodes}


# --------------------------------------------------------------------------- #
# Core star-merge behavior
# --------------------------------------------------------------------------- #


def test_star_burst_contracts_all_candidate_children_at_once() -> None:
    graph = star_graph("A", "B", 4, prefix="s")
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit([graph])

    token = edge_star_token(0)
    assert coarsener.encoder_.edge_rules[0].token == token
    assert coarsener.history_[0]["token"] == token
    assert coarsener.history_[0]["count"] == 4
    assert coarsener.history_[0]["actual_events"] == 1
    assert coarsener.history_[0]["children_absorbed"] == 4

    encoded = coarsener.transform(graph)
    # All five raw nodes collapse into a single star node.
    assert encoded.number_of_nodes() == 1
    assert [data["label"] for _, data in encoded.nodes(data=True)] == [token]
    assert all(
        set(data) == EXPECTED_ENCODED_NODE_FIELDS
        for _, data in encoded.nodes(data=True)
    )

    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


def test_star_merge_differs_from_plain_edge_bpe() -> None:
    graph = star_graph("A", "B", 4, prefix="s")

    star = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit([graph])
    plain = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit([graph])

    star_encoded = star.transform(graph)
    plain_encoded = plain.transform(graph)

    # The star variant absorbs every matching child in one merge; plain edge BPE
    # contracts a single non-overlapping pair and leaves the other children.
    assert star_encoded.number_of_nodes() == 1
    assert plain_encoded.number_of_nodes() > star_encoded.number_of_nodes()

    # Both remain lossless.
    assert uid_edge_set(star.decode(star_encoded)) == uid_edge_set(graph)
    assert uid_edge_set(plain.decode(plain_encoded)) == uid_edge_set(graph)


def test_mixed_arity_groups_share_one_token() -> None:
    # Two parents under the same pair absorb different numbers of children (3 and
    # 2).  Arity is ignored, so both contracted nodes carry the same label.
    graph = nx.DiGraph()
    graph.add_node("r", label="R", time=0.0, uid="r")
    for parent_name, k in [("P1", 3), ("P2", 2)]:
        graph.add_node(parent_name, label="P", time=1.0, uid=parent_name)
        graph.add_edge("r", parent_name)
        for j in range(k):
            child = f"{parent_name}_c{j}"
            graph.add_node(child, label="C", time=2.0 + j, uid=child)
            graph.add_edge(parent_name, child)

    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit([graph])

    token = edge_star_token(0)
    assert coarsener.encoder_.edge_rules[0].token == token
    assert coarsener.history_[0]["count"] == 5
    assert coarsener.history_[0]["actual_events"] == 2
    assert coarsener.history_[0]["children_absorbed"] == 5

    encoded = coarsener.transform(graph)
    contracted_labels = [
        data["label"]
        for _, data in encoded.nodes(data=True)
        if data["label"] == token
    ]
    # Both contracted parents share the one rank token despite differing arity.
    assert contracted_labels == [token, token]

    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)


def test_single_child_merge_uses_rank_token() -> None:
    trees = [
        path_graph(["A", "B", "C"], prefix="a"),
        path_graph(["A", "B", "D"], prefix="b"),
    ]
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit(trees)

    token = edge_star_token(0)
    rule = coarsener.encoder_.edge_rules[0]
    assert rule.token == token
    assert (rule.parent_label, rule.child_label) == ("A", "B")
    assert coarsener.history_[0]["children_absorbed"] == 2

    encoded = coarsener.transform(trees[0])
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(trees[0])


# --------------------------------------------------------------------------- #
# Attachment / grandchild reattachment
# --------------------------------------------------------------------------- #


def test_multilevel_star_merge_reattaches_grandchildren_and_roundtrips() -> None:
    graph = nx.DiGraph()
    graph.add_node("A", label="A", time=0.0, uid="A")
    for i in range(3):
        b = f"B{i}"
        graph.add_node(b, label="B", time=1.0 + i, uid=b)
        graph.add_edge("A", b)
        for j, gl in enumerate(["X", "Y"]):
            g = f"{b}_{gl}"
            graph.add_node(g, label=gl, time=10.0 + j, uid=g)
            graph.add_edge(b, g)

    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=5, min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


def test_exact_type_records_one_component_per_contracted_child() -> None:
    graph = star_graph("A", "B", 3, prefix="s")
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)

    (exact,) = [
        data["type"]
        for _, data in encoded.nodes(data=True)
        if isinstance(data["type"], CompositeType)
    ]
    # One parent component plus three contracted children.
    assert exact.n_components == 4
    assert exact.parent == (-1, 0, 0, 0)
    assert exact.component_labels == ("A", "B", "B", "B")
    assert exact.root_count == 1
    assert exact.site_count == 4


def test_self_pair_chain_roundtrips() -> None:
    # A chain of identical labels exercises overlapping parent/child groups.
    graph = path_graph(["A", "A", "A", "A", "A"], prefix="chain")
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=4, min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


# --------------------------------------------------------------------------- #
# Fitting behavior
# --------------------------------------------------------------------------- #


def test_fit_is_deterministic() -> None:
    graph = star_graph("A", "B", 4, prefix="s")
    first = EdgeBPEWithAutoStarCoarsener(num_merges=3, min_pair_count=2).fit([graph])
    second = EdgeBPEWithAutoStarCoarsener(num_merges=3, min_pair_count=2).fit([graph])
    assert [h["token"] for h in first.history_] == [h["token"] for h in second.history_]

    first_encoded = first.transform(graph)
    second_encoded = second.transform(graph)
    assert [d["label"] for _, d in first_encoded.nodes(data=True)] == [
        d["label"] for _, d in second_encoded.nodes(data=True)
    ]


def test_transform_contracts_any_arity_with_learned_pair() -> None:
    # Fit on arity-2 stars only, then transform a fresh arity-3 star.  Arity is
    # ignored, so the learned (A, B) rule still contracts all three children at
    # once -- star-condensed nodes are identified by label alone.
    train = [
        star_graph("A", "B", 2, prefix="t1"),
        star_graph("A", "B", 2, prefix="t2"),
    ]
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit(train)
    token = edge_star_token(0)
    assert coarsener.encoder_.edge_rules[0].token == token

    fresh = star_graph("A", "B", 3, prefix="u")
    encoded = coarsener.transform(fresh)
    # All three children contract into the single parent regardless of arity.
    assert encoded.number_of_nodes() == 1
    assert [data["label"] for _, data in encoded.nodes(data=True)] == [token]
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(fresh)


def test_min_pair_count_blocks_singletons() -> None:
    # A single A->B edge has count 1, below the default threshold of 2.
    graph = path_graph(["A", "B"], prefix="x")
    coarsener = EdgeBPEWithAutoStarCoarsener(min_pair_count=2).fit([graph])
    assert coarsener.history_ == []
    assert coarsener.encoder_.edge_rules == ()

    encoded = coarsener.transform(graph)
    assert encoded.number_of_nodes() == 2
    assert uid_edge_set(coarsener.decode(encoded)) == uid_edge_set(graph)


def test_num_merges_caps_learned_rules() -> None:
    trees = [
        path_graph(["A", "B", "C", "D"], prefix="a"),
        path_graph(["A", "B", "C", "D"], prefix="b"),
    ]
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=1, min_pair_count=2).fit(trees)
    assert len(coarsener.history_) == 1
    assert uid_edge_set(coarsener.decode(coarsener.transform(trees[0]))) == uid_edge_set(
        trees[0]
    )


@pytest.mark.parametrize("pair_score", ["count", "normalized", "size_weighted"])
def test_pair_score_options_roundtrip(pair_score: str) -> None:
    trees = [
        star_graph("A", "B", 3, prefix="g1"),
        path_graph(["A", "B", "B"], prefix="g2"),
    ]
    coarsener = EdgeBPEWithAutoStarCoarsener(
        num_merges=2, min_pair_count=2, pair_score=pair_score
    ).fit(trees)
    assert coarsener.pair_score_display_name_ == pair_score
    for tree in trees:
        decoded = coarsener.decode(coarsener.transform(tree))
        assert uid_edge_set(decoded) == uid_edge_set(tree)


def test_custom_pair_score_callable_is_accepted() -> None:
    def first_only(n_ab, n_a, n_b, s_a, s_b):  # noqa: ANN001
        return float(n_ab)

    graph = star_graph("A", "B", 3, prefix="s")
    coarsener = EdgeBPEWithAutoStarCoarsener(
        num_merges=1, min_pair_count=2, pair_score=first_only
    ).fit([graph])
    assert coarsener.pair_score_name_ is None
    assert uid_edge_set(coarsener.decode(coarsener.transform(graph))) == uid_edge_set(graph)


def test_invalid_pair_score_name_rejected() -> None:
    with pytest.raises(ValueError):
        EdgeBPEWithAutoStarCoarsener(pair_score="nonexistent")


def test_multiple_graphs_transform_preserves_container_shape() -> None:
    trees = [
        star_graph("A", "B", 3, prefix="a"),
        star_graph("A", "B", 2, prefix="b"),
    ]
    coarsener = EdgeBPEWithAutoStarCoarsener(num_merges=2, min_pair_count=2).fit(trees)
    encoded = coarsener.transform(trees)
    assert isinstance(encoded, list) and len(encoded) == 2
    decoded = coarsener.decode(encoded)
    for original, restored in zip(trees, decoded):
        assert uid_edge_set(restored) == uid_edge_set(original)


def test_edge_star_rule_stores_single_token() -> None:
    rule = EdgeStarRule(
        rank=0,
        token=edge_star_token(0),
        parent_label="A",
        child_label="B",
        count=2,
    )
    assert rule.token == ("edge_star", 0)
    assert rule.parent_token == "A"
    assert rule.child_token == "B"


def test_edge_star_token_is_rank_only() -> None:
    assert edge_star_token(0) == ("edge_star", 0)
    assert edge_star_token(3) == ("edge_star", 3)
