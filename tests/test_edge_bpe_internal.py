from __future__ import annotations

from collections import Counter, defaultdict
import random

import networkx as nx

from tree_coarsening import TokenSpec, Vocabulary, edge_bpe_token
from tree_coarsening.coarseners.edge_bpe import _CompactEdgeTree, _TokenCodec


def _recount(state: _CompactEdgeTree) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    index: dict = defaultdict(set)
    for child in range(len(state.parent)):
        if state._edge_is_live(child):
            key = state._edge_key(child)
            counts[key] += 1
            index[key].add(child)
    return counts, dict(index)


def test_incremental_label_pair_counts_match_full_recounts() -> None:
    for seed in range(12):
        rng = random.Random(seed)
        graph = nx.DiGraph()
        n_nodes = rng.randint(25, 70)
        labels = ("A", "B", "C")
        for node in range(n_nodes):
            graph.add_node(
                node,
                label=rng.choice(labels),
                type=("base", rng.choice(labels)),  # overwritten below for consistency
                size=1,
                time=float(rng.randrange(12)),
                uid=(seed, node),
                super_label=(seed, node),
                super_uids=((seed, node),),
            )
            graph.nodes[node]["type"] = ("base", graph.nodes[node]["label"])
            if node:
                graph.add_edge(rng.randrange(node), node, attach_map=(0,))

        vocab = Vocabulary(
            symbols={label: TokenSpec(site_count=1, root_count=1) for label in labels}
        )
        codec = _TokenCodec()
        counts: Counter = Counter()
        state = _CompactEdgeTree.from_graph(
            graph,
            codec=codec,
            vocab=vocab,
            pair_counts=counts,
            capture_output=False,
        )
        original_length = len(state.parent)

        for rank in range(24):
            recounted, rebuilt_index = _recount(state)
            assert counts == recounted
            assert {key: set(value) for key, value in state.edge_index.items()} == rebuilt_index
            assert len(state.parent) == original_length
            assert state.output is None

            if not counts:
                break
            key = rng.choice(tuple(counts))
            parent_id, child_id = key
            parent_label = codec.decode(parent_id)
            child_label = codec.decode(child_id)
            token = edge_bpe_token(rank)
            vocab.add_symbol(
                token,
                TokenSpec(
                    site_count=vocab.site_count(parent_label) + vocab.site_count(child_label),
                    root_count=vocab.root_count(parent_label),
                ),
            )
            assert state.contract_pair(
                key,
                new_label=codec.intern(token),
                pair_counts=counts,
            ) > 0

        recounted, rebuilt_index = _recount(state)
        assert counts == recounted
        assert {key: set(value) for key, value in state.edge_index.items()} == rebuilt_index
        assert len(state.parent) == original_length
