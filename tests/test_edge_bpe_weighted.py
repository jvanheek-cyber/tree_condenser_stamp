from __future__ import annotations

import math

import networkx as nx
import pytest

from tree_coarsening import CompositeType, EdgeBPECoarsener, base_token
from tree_coarsening.schema import RAW_INPUT_FLAG


def edge_graph(parent_label: str, child_label: str, *, prefix: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label=parent_label, time=0.0, uid=f"{prefix}p")
    graph.add_node(1, label=child_label, time=1.0, uid=f"{prefix}c")
    graph.add_edge(0, 1)
    return graph


def singleton_graph(label: str, *, uid: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label=label, time=0.0, uid=uid)
    return graph


def encoded_edge_graph(
    parent_label: str,
    child_label: str,
    *,
    parent_size: int,
    child_size: int,
    prefix: str,
) -> nx.DiGraph:
    """Construct a valid prior-stage graph with chosen fitting-label sizes."""

    def node_payload(label: str, size: int, stem: str):
        uids = tuple(f"{stem}{i}" for i in range(size))
        if size == 1:
            exact_type = base_token(label)
            super_label = uids[0]
        else:
            exact_type = CompositeType(
                model_id="weighted-test-stage",
                kind="component",
                label=label,
                parent=(-1,) + (0,) * (size - 1),
                component_labels=tuple(f"{label}:{i}" for i in range(size)),
                component_types=tuple(base_token(f"{label}:{i}") for i in range(size)),
                component_sizes=(1,) * size,
                component_root_counts=(1,) * size,
                attach=(0,) * (size - 1),
            )
            super_label = uids
        return {
            "label": label,
            "type": exact_type,
            "size": size,
            "time": float(size),
            "super_label": super_label,
            "super_uids": uids,
        }

    graph = nx.DiGraph()
    graph.add_node(0, **node_payload(parent_label, parent_size, f"{prefix}p"))
    graph.add_node(1, **node_payload(child_label, child_size, f"{prefix}c"))
    graph.add_edge(0, 1, attach_map=(0,))
    graph.graph[RAW_INPUT_FLAG] = False
    return graph


def path_graph(labels: list[str], *, prefix: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i, label in enumerate(labels):
        graph.add_node(i, label=label, time=float(i), uid=f"{prefix}{i}")
        if i:
            graph.add_edge(i - 1, i)
    return graph


def test_weighted_pair_scores_change_rule_selection() -> None:
    # Raw count prefers A->B: four edges versus three C->D edges. Extra A and
    # B singleton occurrences reduce only the normalized A->B score.
    normalized_corpus = [
        *(edge_graph("A", "B", prefix=f"ab{i}") for i in range(4)),
        *(singleton_graph("A", uid=f"extra-a-{i}") for i in range(16)),
        *(singleton_graph("B", uid=f"extra-b-{i}") for i in range(16)),
        *(edge_graph("C", "D", prefix=f"cd{i}") for i in range(3)),
    ]

    count_model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        pair_score="count",
    ).fit(normalized_corpus)
    normalized_model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        pair_score="normalized",
    ).fit(normalized_corpus)

    assert (
        count_model.history_[0]["parent_label"],
        count_model.history_[0]["child_label"],
    ) == ("A", "B")
    first = normalized_model.history_[0]
    assert (first["parent_label"], first["child_label"]) == ("C", "D")
    assert first["count"] == 3
    assert first["parent_count"] == first["child_count"] == 3
    assert first["score"] == pytest.approx(1.0)

    # Raw count again prefers A->B, but the three X->Y edges combine symbols
    # ten times larger, so size weighting selects X->Y.
    size_corpus = [
        *(
            encoded_edge_graph(
                "A", "B", parent_size=1, child_size=1, prefix=f"ab{i}"
            )
            for i in range(4)
        ),
        *(
            encoded_edge_graph(
                "X", "Y", parent_size=10, child_size=10, prefix=f"xy{i}"
            )
            for i in range(3)
        ),
    ]
    size_weighted_model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        pair_score="size_weighted",
    ).fit(size_corpus)

    first = size_weighted_model.history_[0]
    assert (first["parent_label"], first["child_label"]) == ("X", "Y")
    assert (first["parent_size"], first["child_size"]) == (10, 10)
    assert first["score"] == pytest.approx(60.0)


def test_custom_pair_score_and_incremental_label_counts() -> None:
    seen_arguments: list[tuple[int, int, int, int, int]] = []

    def scorer(n_ab: int, n_a: int, n_b: int, s_a: int, s_b: int) -> float:
        seen_arguments.append((n_ab, n_a, n_b, s_a, s_b))
        return n_ab + 0.01 * (s_a + s_b) - 0.001 * (n_a + n_b)

    model = EdgeBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        pair_score=scorer,
    ).fit([path_graph(["A", "A", "A", "A"], prefix="same")])

    assert seen_arguments
    first, second = model.history_[:2]
    assert first["parent_label"] == first["child_label"] == "A"
    assert first["parent_count"] == first["child_count"] == 4
    assert first["actual_events"] == 2
    # Two disjoint A->A contractions consume four A occurrences and create two
    # occurrences of the first learned label. The second score sees those two.
    assert second["parent_count"] == second["child_count"] == 2
    assert first["pair_score"] == "scorer"


def test_pair_score_must_be_finite() -> None:
    model = EdgeBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        pair_score=lambda n_ab, n_a, n_b, s_a, s_b: math.nan,
    )
    with pytest.raises(Exception, match="non-finite"):
        model.fit([edge_graph("A", "B", prefix="bad")])


@pytest.mark.parametrize("pair_score", ["count", "normalized", "size_weighted"])
def test_numba_and_python_weighted_histories_match(pair_score: str) -> None:
    pytest.importorskip("numba")
    corpus = [
        *(edge_graph("A", "B", prefix=f"ab{i}") for i in range(5)),
        *(singleton_graph("A", uid=f"a{i}") for i in range(4)),
        *(singleton_graph("B", uid=f"b{i}") for i in range(4)),
        *(edge_graph("C", "D", prefix=f"cd{i}") for i in range(3)),
    ]
    kwargs = {
        "num_merges": 3,
        "min_pair_count": 1,
        "pair_score": pair_score,
        "model_id": "weighted-parity",
    }
    python_model = EdgeBPECoarsener(backend="python", **kwargs).fit(corpus)
    numba_model = EdgeBPECoarsener(backend="numba", **kwargs).fit(corpus)
    assert numba_model.history_ == python_model.history_
