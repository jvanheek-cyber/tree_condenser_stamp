from __future__ import annotations

from collections import Counter, defaultdict
import random

import networkx as nx

from tree_coarsening.coarseners.edge_bpe import (
    _CompactEdgeTree,
    _TokenCodec,
    edge_bpe_token,
)
from tree_coarsening.vocabulary import VocabEntry, Vocabulary


def _recount(state: _CompactEdgeTree) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    index: dict = defaultdict(set)
    for child in range(len(state.parent)):
        if state._edge_is_live(child):
            key = state._edge_key(child)
            counts[key] += 1
            index[key].add(child)
    return counts, dict(index)


def test_incremental_edge_counts_match_full_recounts() -> None:
    for seed in range(12):
        rng = random.Random(seed)
        graph = nx.DiGraph()
        n_nodes = rng.randint(25, 70)
        for node in range(n_nodes):
            graph.add_node(
                node,
                label=rng.choice(("A", "B", "C")),
                time=float(rng.randrange(12)),
                uid=(seed, node),
            )
            if node:
                graph.add_edge(rng.randrange(node), node)

        vocab = Vocabulary()
        codec = _TokenCodec()
        pair_counts: Counter = Counter()
        state = _CompactEdgeTree.from_raw_graph(
            graph,
            codec=codec,
            vocab=vocab,
            pair_counts=pair_counts,
            capture_output=False,
        )
        original_array_length = len(state.parent)

        for rank in range(24):
            recounted, rebuilt_index = _recount(state)
            assert pair_counts == recounted
            assert {key: set(value) for key, value in state.edge_index.items()} == rebuilt_index
            assert len(state.parent) == original_array_length
            assert state.output is None
            assert state.uid_ref is None

            if not pair_counts:
                break
            key = rng.choice(tuple(pair_counts))
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
            assert state.contract_and_count_pairs(
                key,
                new_label=new_id,
                pair_counts=pair_counts,
            ) > 0

        recounted, rebuilt_index = _recount(state)
        assert pair_counts == recounted
        assert {key: set(value) for key, value in state.edge_index.items()} == rebuilt_index
        assert len(state.parent) == original_array_length
