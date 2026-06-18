from __future__ import annotations

import random

import networkx as nx
import pytest

pytest.importorskip("numba")

from tree_coarsening import EdgeBPECoarsener
from tree_coarsening.coarseners.edge_bpe import (
    _CompactEdgeTree,
    _TokenCodec,
    edge_bpe_token,
)
from tree_coarsening.coarseners.edge_bpe_numba import NumbaTrainingForest
from tree_coarsening.vocabulary import VocabEntry, Vocabulary


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


def test_numba_backend_matches_python_rule_history_and_roundtrips() -> None:
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
    encoded = numba_model.transform(graphs[0])
    decoded = numba_model.decode(encoded)
    assert _uid_edges(decoded) == _uid_edges(graphs[0])


def test_numba_incremental_counts_match_full_recount() -> None:
    graph = _random_tree(41, n_nodes=120)
    vocab = Vocabulary()
    codec = _TokenCodec()
    state = _CompactEdgeTree.from_raw_graph(
        graph,
        codec=codec,
        vocab=vocab,
        capture_output=False,
        build_edge_index=False,
    )
    forest = NumbaTrainingForest.from_compact_states([state], label_capacity=80)

    for rank in range(20):
        forest.assert_counts_match_recount()
        best = forest.select_best_pair(1, codec)
        if best is None:
            break
        key, _ = best
        parent_id, child_id, attach_site = key
        token = edge_bpe_token(rank)
        vocab.add(
            VocabEntry(
                token=token,
                parent=(-1, 0),
                label=(codec.decode(parent_id), codec.decode(child_id)),
                attach=(attach_site,),
                created_at_step=rank,
                operation="edge",
            )
        )
        new_id = codec.intern(token)
        forest.register_label(new_id, vocab.site_count(token))
        assert forest.contract_pair(key, new_label=new_id) > 0

    forest.assert_counts_match_recount()
