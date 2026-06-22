from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from tree_coarsening import (
    EdgeBPECoarsener,
    GreedyStarBPECoarsener,
    GreedyStarBPEEncoder,
    edge_bpe_token,
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


def two_star_graph() -> nx.DiGraph:
    """Root with two equal-frequency stars under children ``A`` and ``C``."""

    graph = nx.DiGraph()
    graph.add_node("r", label="R", time=0.0, uid="r")
    for parent, child in [("A", "B"), ("C", "D")]:
        graph.add_node(parent, label=parent, time=1.0, uid=parent)
        graph.add_edge("r", parent)
        for j in range(3):
            leaf = f"{child}{j}"
            graph.add_node(leaf, label=child, time=2.0 + j, uid=leaf)
            graph.add_edge(parent, leaf)
    return graph


def uid_edge_set(graph: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (graph.nodes[u].get("uid", u), graph.nodes[v].get("uid", v))
        for u, v in graph.edges
    }


def uid_node_set(graph: nx.DiGraph) -> set[str]:
    return {graph.nodes[n].get("uid", n) for n in graph.nodes}


# --------------------------------------------------------------------------- #
# Greedy chaining behavior
# --------------------------------------------------------------------------- #


def test_greedy_chains_same_child_off_new_token() -> None:
    graph = star_graph("A", "B", 4, prefix="s")
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])

    history = coarsener.history_
    # First merge is the ordinary global pick; the rest chain off the new token.
    assert history[0]["parent_label"] == "A"
    assert history[0]["greedy"] is False
    assert history[0]["count"] == 4
    for prev, curr in zip(history, history[1:]):
        assert curr["greedy"] is True
        assert curr["parent_label"] == prev["token"]
        assert curr["child_label"] == "B"

    # min_pair_count gates only the chain start; the greedy chain runs to the
    # final child, so all four children of the star are consumed.
    assert [h["token"] for h in history] == [edge_bpe_token(i) for i in range(4)]
    assert [h["count"] for h in history] == [4, 3, 2, 1]


def test_min_pair_count_one_consumes_whole_star() -> None:
    graph = star_graph("A", "B", 4, prefix="s")
    coarsener = GreedyStarBPECoarsener(min_pair_count=1).fit([graph])

    encoded = coarsener.transform(graph)
    assert encoded.number_of_nodes() == 1
    assert [d["label"] for _, d in encoded.nodes(data=True)] == [edge_bpe_token(3)]
    assert all(
        set(d) == EXPECTED_ENCODED_NODE_FIELDS for _, d in encoded.nodes(data=True)
    )

    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


def test_greedy_finishes_a_star_before_switching() -> None:
    graph = two_star_graph()
    greedy = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])
    plain = EdgeBPECoarsener(min_pair_count=2).fit([graph])

    greedy_children = [h["child_label"] for h in greedy.history_]
    plain_children = [h["child_label"] for h in plain.history_]

    # Greedy exhausts one star's chain before touching the other; plain BPE
    # interleaves the two stars by global frequency.
    assert greedy_children[0] == greedy_children[1]
    assert plain_children[0] != plain_children[1]


def test_roundtrip_is_lossless() -> None:
    graph = two_star_graph()
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


# --------------------------------------------------------------------------- #
# max_steps counts whole star-chains
# --------------------------------------------------------------------------- #


def test_encoder_emits_greedy_encoder_with_group_starts() -> None:
    graph = two_star_graph()
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])
    encoder = coarsener.encoder_
    assert isinstance(encoder, GreedyStarBPEEncoder)
    # group_starts aligns with the learned rules; True marks a new chain.
    assert len(encoder.group_starts) == len(encoder.edge_rules)
    assert encoder.group_starts == tuple(
        not h["greedy"] for h in coarsener.history_
    )
    # Two stars -> exactly two chain starts.
    assert sum(encoder.group_starts) == 2


def test_max_steps_one_applies_a_whole_star_chain() -> None:
    graph = two_star_graph()
    # min_pair_count=1 makes each chain fully consume its star, so a completed
    # chain leaves a single merged node with no leftover leaves.
    coarsener = GreedyStarBPECoarsener(min_pair_count=1).fit([graph])
    encoder = coarsener.encoder_

    full = encoder.encode(graph)
    partial = encoder.encode(graph, max_steps=1)

    # The first chain has three rules (3 children -> 3 merges).  max_steps=1 must
    # apply all of them, fully condensing the first star and leaving the second
    # star completely untouched -- never a partially merged star.
    first_chain_len = next(
        i for i, start in enumerate(encoder.group_starts[1:], start=1) if start
    )
    assert first_chain_len == 3
    # One node removed per applied merge rule.
    assert partial.number_of_nodes() == graph.number_of_nodes() - first_chain_len

    # The remaining star is fully intact: three identical leaves of one label.
    leaf_labels = [
        d["label"]
        for _, d in partial.nodes(data=True)
        if d["label"] in {"B", "D"}
    ]
    assert len(leaf_labels) == 3 and len(set(leaf_labels)) == 1

    # Full encode collapses both stars further than the single-chain encode.
    assert full.number_of_nodes() < partial.number_of_nodes()


def test_max_steps_zero_applies_nothing() -> None:
    graph = two_star_graph()
    encoder = GreedyStarBPECoarsener(min_pair_count=1).fit([graph]).encoder_
    encoded = encoder.encode(graph, max_steps=0)
    assert encoded.number_of_nodes() == graph.number_of_nodes()


def test_max_steps_beyond_chain_count_applies_all() -> None:
    graph = two_star_graph()
    encoder = GreedyStarBPECoarsener(min_pair_count=1).fit([graph]).encoder_
    full = encoder.encode(graph)
    capped = encoder.encode(graph, max_steps=999)
    assert capped.number_of_nodes() == full.number_of_nodes()


def test_multilevel_star_reattaches_grandchildren_and_roundtrips() -> None:
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

    coarsener = GreedyStarBPECoarsener(num_merges=10, min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


# --------------------------------------------------------------------------- #
# Straightforward-BPE encoding semantics
# --------------------------------------------------------------------------- #


def test_transform_applies_chain_to_fresh_data() -> None:
    train = [
        star_graph("A", "B", 4, prefix="t1"),
        star_graph("A", "B", 4, prefix="t2"),
    ]
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit(train)

    fresh = star_graph("A", "B", 4, prefix="u")
    encoded = coarsener.transform(fresh)
    # The chained rules apply as ordinary BPE merges, eating one child each.
    n_rules = len(coarsener.encoder_.edge_rules)
    assert encoded.number_of_nodes() == fresh.number_of_nodes() - n_rules
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(fresh)


def test_single_edge_below_threshold_is_untouched() -> None:
    graph = path_graph(["A", "B"], prefix="x")
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])
    assert coarsener.history_ == []
    assert coarsener.encoder_.edge_rules == ()

    encoded = coarsener.transform(graph)
    assert encoded.number_of_nodes() == 2
    assert uid_edge_set(coarsener.decode(encoded)) == uid_edge_set(graph)


def test_self_pair_chain_roundtrips() -> None:
    graph = path_graph(["A", "A", "A", "A", "A"], prefix="chain")
    coarsener = GreedyStarBPECoarsener(num_merges=4, min_pair_count=1).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


# --------------------------------------------------------------------------- #
# Configuration / parity with edge BPE
# --------------------------------------------------------------------------- #


def test_fit_is_deterministic() -> None:
    graph = two_star_graph()
    first = GreedyStarBPECoarsener(num_merges=3, min_pair_count=2).fit([graph])
    second = GreedyStarBPECoarsener(num_merges=3, min_pair_count=2).fit([graph])
    assert [h["token"] for h in first.history_] == [h["token"] for h in second.history_]


def test_num_merges_caps_learned_rules() -> None:
    graph = star_graph("A", "B", 5, prefix="s")
    coarsener = GreedyStarBPECoarsener(num_merges=2, min_pair_count=2).fit([graph])
    assert len(coarsener.history_) == 2
    assert uid_edge_set(coarsener.decode(coarsener.transform(graph))) == uid_edge_set(
        graph
    )


@pytest.mark.parametrize("pair_score", ["count", "normalized", "size_weighted"])
def test_pair_score_options_roundtrip(pair_score: str) -> None:
    graph = two_star_graph()
    coarsener = GreedyStarBPECoarsener(
        num_merges=4, min_pair_count=2, pair_score=pair_score
    ).fit([graph])
    assert coarsener.pair_score_display_name_ == pair_score
    decoded = coarsener.decode(coarsener.transform(graph))
    assert uid_edge_set(decoded) == uid_edge_set(graph)


def test_custom_pair_score_callable_is_accepted() -> None:
    def by_count(n_ab, n_a, n_b, s_a, s_b):  # noqa: ANN001
        return float(n_ab)

    graph = star_graph("A", "B", 4, prefix="s")
    coarsener = GreedyStarBPECoarsener(
        min_pair_count=2, pair_score=by_count
    ).fit([graph])
    assert coarsener.pair_score_name_ is None
    assert uid_edge_set(coarsener.decode(coarsener.transform(graph))) == uid_edge_set(
        graph
    )


def test_invalid_pair_score_name_rejected() -> None:
    with pytest.raises(ValueError):
        GreedyStarBPECoarsener(pair_score="nonexistent")


def test_numba_backend_rejected() -> None:
    with pytest.raises(ValueError):
        GreedyStarBPECoarsener(backend="numba")


def test_multiple_graphs_transform_preserves_container_shape() -> None:
    trees = [
        star_graph("A", "B", 3, prefix="a"),
        star_graph("A", "B", 4, prefix="b"),
    ]
    coarsener = GreedyStarBPECoarsener(num_merges=3, min_pair_count=2).fit(trees)
    encoded = coarsener.transform(trees)
    assert isinstance(encoded, list) and len(encoded) == 2
    decoded = coarsener.decode(encoded)
    for original, restored in zip(trees, decoded):
        assert uid_edge_set(restored) == uid_edge_set(original)


# --------------------------------------------------------------------------- #
# Regression: a node must never keep a child whose subtree is *identical* to a
# subtree already merged inside that node.
#
# Symptom (once reported): a parent subtree contained, as one of its own
# children, a subtree equal to a component the parent already absorbed.  It
# reproduced on a tree from the *training* set, at full encode, with no
# ``max_steps`` involved.
#
# Root cause (fixed): the greedy chain used to extend by appending only the
# *same single child label* last merged, recognizing a "star" of identical
# children only when they shared one scalar edge label.  When a node accumulated
# a *compound* subtree by absorbing its pieces one base label at a time, a
# sibling that was the very same compound subtree had already been condensed into
# a single component token, which the raw-label chain could not consume -- so it
# was stranded as a duplicate.  The fix extends the chain to absorb any child
# whose label is structurally one of the new token's merged components (its
# merge-closure), so condensed siblings are folded into the same chain.
# --------------------------------------------------------------------------- #


def _subtree_signature(node_type: object) -> tuple:
    """Canonical structural signature of a node's full (expanded) subtree."""

    from tree_coarsening import CompositeType

    if isinstance(node_type, CompositeType):
        return (
            "composite",
            node_type.label,
            tuple(_subtree_signature(component) for component in node_type.component_types),
        )
    return ("base", node_type)


def duplicate_child_subtree_edges(graph: nx.DiGraph) -> list[tuple[Any, Any]]:
    """Edges whose child subtree also appears inside the parent's merged type.

    Returns ``(parent, child)`` node pairs where the child's full subtree is
    structurally identical to one of the components already merged into the
    parent -- the reported invariant violation.
    """

    from tree_coarsening import CompositeType

    offenders: list[tuple[Any, Any]] = []
    for parent, child in graph.edges:
        parent_type = graph.nodes[parent]["type"]
        if not isinstance(parent_type, CompositeType):
            continue
        child_signature = _subtree_signature(graph.nodes[child]["type"])
        component_signatures = [
            _subtree_signature(component) for component in parent_type.component_types
        ]
        if child_signature in component_signatures:
            offenders.append((parent, child))
    return offenders


def repeated_compound_subtree_tree() -> nx.DiGraph:
    """A 9-node tree that strands a condensed compound sibling on self-encode.

    Shape (node ids in parentheses)::

        B(0)
          A(1)
            C(2)
              C(3)
                A(4)
              C(5)
                C(6)
                A(7)
              A(8)

    Fitting on this single tree learns the chain ``(C,C) -> (T0,A) -> (T1,A)``.
    Encoding it back folds ``C(2)`` up to a ``T2`` node by absorbing its ``C``
    and ``A`` children one base label at a time, reconstructing a ``T1``-shaped
    subtree inside itself.  Meanwhile ``C(5)`` -- structurally identical to that
    ``T1`` -- has already been condensed to a single ``T1`` token, which the
    ``(T1, A)`` chain cannot absorb (it expects a raw ``A`` child).  The ``T1``
    sibling is therefore left attached to the ``T2`` parent that already
    contains a ``T1`` component.
    """

    graph = nx.DiGraph()

    def add(node_id: int, label: str) -> None:
        graph.add_node(node_id, label=label, time=float(node_id), uid=f"n{node_id}")

    for node_id, label in enumerate("BACCACCAA"):
        add(node_id, label)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 5)
    graph.add_edge(5, 6)
    graph.add_edge(5, 7)
    graph.add_edge(2, 8)
    return graph


def test_simple_star_has_no_duplicate_child_subtree() -> None:
    """A single-level star of identical scalar children collapses cleanly."""

    train = [star_graph("A", "B", 3, prefix="t1"), star_graph("A", "B", 3, prefix="t2")]
    encoder = GreedyStarBPECoarsener(min_pair_count=2).fit(train).encoder_

    same_size = star_graph("A", "B", 3, prefix="x")
    encoded = encoder.encode(same_size)
    assert duplicate_child_subtree_edges(encoded) == []


def test_max_steps_does_not_strand_subtrees_on_simple_star() -> None:
    """``max_steps`` never produces a partial star on a clean single-level star.

    Encoding a within-distribution star is clean for *every* number of chain
    steps, because ``max_steps`` only cuts between complete chains.  This
    exonerates ``max_steps`` as the source of the duplicate-subtree symptom.
    """

    train = [star_graph("A", "B", 3, prefix="t1"), star_graph("A", "B", 3, prefix="t2")]
    encoder = GreedyStarBPECoarsener(min_pair_count=2).fit(train).encoder_
    n_groups = sum(encoder.group_starts)

    same_size = star_graph("A", "B", 3, prefix="x")
    for steps in range(n_groups + 2):
        encoded = encoder.encode(same_size, max_steps=steps)
        assert duplicate_child_subtree_edges(encoded) == [], (
            f"unexpected duplicate child subtree at max_steps={steps}"
        )


def test_repeated_compound_subtree_children_are_fully_merged() -> None:
    """No node keeps a child equal to a component it already absorbed.

    Regression for the reported bug on a *training* tree at full encode (no
    ``max_steps``): the greedy chain now absorbs the condensed ``T1`` sibling
    instead of stranding it under a ``T2`` parent that already merged a ``T1``
    component.
    """

    graph = repeated_compound_subtree_tree()
    encoder = GreedyStarBPECoarsener(min_pair_count=2).fit([graph]).encoder_

    encoded = encoder.encode(graph)
    assert duplicate_child_subtree_edges(encoded) == []


def test_repeated_compound_subtree_collapses_at_every_max_steps() -> None:
    """The compound star folds into one chain, so no ``max_steps`` strands it.

    The condensed sibling is absorbed within the same chain group as the rest of
    the star, so cutting between complete chains never leaves a duplicate child.
    """

    graph = repeated_compound_subtree_tree()
    encoder = GreedyStarBPECoarsener(min_pair_count=2).fit([graph]).encoder_
    n_groups = sum(encoder.group_starts)

    for steps in range(n_groups + 2):
        encoded = encoder.encode(graph, max_steps=steps)
        assert duplicate_child_subtree_edges(encoded) == [], (
            f"unexpected duplicate child subtree at max_steps={steps}"
        )
    assert duplicate_child_subtree_edges(encoder.encode(graph)) == []



def test_repeated_compound_subtree_roundtrips() -> None:
    """Folding the condensed sibling into the chain stays lossless.

    The fix only changes how aggressively the chain merges; decoding still
    reconstructs the original tree exactly.
    """

    graph = repeated_compound_subtree_tree()
    coarsener = GreedyStarBPECoarsener(min_pair_count=2).fit([graph])
    encoded = coarsener.transform(graph)
    decoded = coarsener.decode(encoded)
    assert uid_edge_set(decoded) == uid_edge_set(graph)
    assert uid_node_set(decoded) == uid_node_set(graph)


