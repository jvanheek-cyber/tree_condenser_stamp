from __future__ import annotations

import networkx as nx

from tree_coarsening import EdgeBPECoarsener, base_token, edge_bpe_token


def path_graph(labels: list[str], *, prefix: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(i, label=label, time=float(i), uid=f"{prefix}{i}")
        if i:
            G.add_edge(i - 1, i)
    return G


def uid_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (G.nodes[u].get("uid", u), G.nodes[v].get("uid", v))
        for u, v in G.edges
    }


def test_edge_bpe_learns_edge_token_and_roundtrips() -> None:
    X = [path_graph(["A", "B", "C"], prefix="a"), path_graph(["A", "B", "D"], prefix="b")]
    coarsener = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(X)

    token = edge_bpe_token(0)
    assert token in coarsener.encoder_.vocab.entries
    entry = coarsener.encoder_.vocab.entries[token]
    assert entry.parent == (-1, 0)
    assert entry.label == (base_token("A"), base_token("B"))
    assert entry.attach == (0,)

    H = coarsener.transform(X[0])
    assert list(H.nodes) == list(range(H.number_of_nodes()))
    assert any(data["label"] == token for _, data in H.nodes(data=True))
    assert all(set(data) == {"label", "super_uids"} for _, data in H.nodes(data=True))

    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(X[0])


def test_edge_bpe_records_nonzero_attachment_site_after_nested_merge() -> None:
    X = [path_graph(["A", "B", "C"], prefix="a"), path_graph(["A", "B", "D"], prefix="b")]
    coarsener = EdgeBPECoarsener(num_merges=2, min_pair_count=1).fit(X)

    assert edge_bpe_token(0) in coarsener.encoder_.vocab.entries
    assert edge_bpe_token(1) in coarsener.encoder_.vocab.entries
    second = coarsener.encoder_.vocab.entries[edge_bpe_token(1)]
    assert second.parent == (-1, 0)
    assert edge_bpe_token(0) in second.label
    assert second.attach == (1,)

    H = coarsener.transform(X[0])
    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(X[0])


def test_edge_bpe_partial_decode_roundtrips() -> None:
    G = path_graph(["A", "B", "C"], prefix="x")
    coarsener = EdgeBPECoarsener(num_merges=2, min_pair_count=1).fit([G])
    H = coarsener.transform(G)
    learned_nodes = [n for n, data in H.nodes(data=True) if data["label"] == edge_bpe_token(1)]
    assert learned_nodes

    partially_decoded = coarsener.decode(H, target=learned_nodes[0], by="node", recursive=False)
    assert all(set(data) == {"label", "super_uids"} for _, data in partially_decoded.nodes(data=True))
    assert uid_edge_set(coarsener.decode(partially_decoded)) == uid_edge_set(G)


def test_edge_bpe_scores_raw_edges_before_overlap_filtering() -> None:
    G = nx.DiGraph()
    G.add_node(0, label="A", time=0.0, uid="root")
    for i in range(1, 9):
        G.add_node(i, label="B", time=float(i), uid=f"b{i}")
        G.add_edge(0, i)

    coarsener = EdgeBPECoarsener(num_merges=1, min_pair_count=1).fit([G])
    assert coarsener.history_[0]["count"] == 8
    assert coarsener.history_[0]["count_semantics"] == "raw_matching_edges"
    assert coarsener.history_[0]["actual_events"] == 1


def test_edge_bpe_long_repeated_merge_preserves_uid_order() -> None:
    G = path_graph(["A"] * 256, prefix="p")
    coarsener = EdgeBPECoarsener(num_merges=20, min_pair_count=1).fit([G])
    H = coarsener.transform(G)

    flattened = tuple(uid for _, data in H.nodes(data=True) for uid in data["super_uids"])
    assert set(flattened) == {f"p{i}" for i in range(256)}
    assert len(flattened) == 256
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)
