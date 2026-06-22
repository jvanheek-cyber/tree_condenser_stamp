from __future__ import annotations

import random

import networkx as nx
import pytest

pytest.importorskip("numba")

from tree_coarsening import EdgeBPECoarsener


def _random_tree(seed: int, n_nodes: int = 90) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    for node in range(n_nodes):
        graph.add_node(
            node,
            label=rng.choice(("A", "B", "C")),
            time=float(rng.randrange(20)) + node / 1000,
            uid=(seed, node),
        )
        if node:
            graph.add_edge(rng.randrange(node), node)
    return graph


def _uid_edges(graph: nx.DiGraph) -> set[tuple[object, object]]:
    return {
        (graph.nodes[parent].get("uid", parent), graph.nodes[child].get("uid", child))
        for parent, child in graph.edges
    }


def test_numba_backend_has_identical_semantics() -> None:
    graphs = [_random_tree(seed) for seed in range(3)]
    kwargs = {
        "num_merges": 24,
        "min_pair_count": 2,
        "validate_inputs": False,
        "model_id": "backend-parity",
    }
    python_model = EdgeBPECoarsener(backend="python", **kwargs).fit(graphs)
    numba_model = EdgeBPECoarsener(backend="numba", **kwargs).fit(graphs)

    assert numba_model.history_ == python_model.history_
    assert numba_model.backend_used_ == "numba"
    encoded = numba_model.transform(graphs[0])
    decoded = numba_model.decode(encoded)
    assert _uid_edges(decoded) == _uid_edges(graphs[0])
