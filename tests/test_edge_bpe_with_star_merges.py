from __future__ import annotations

import networkx as nx

from tree_coarsening import (
    EdgeBPEWithStarMergesCoarsener,
    base_token,
    edge_bpe_token,
    edge_star_token,
)


def path_graph(labels: list[str], *, prefix: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(i, label=label, time=float(i), uid=f"{prefix}{i}")
        if i:
            G.add_edge(i - 1, i)
    return G


def star_graph(parent_label: str, child_label: str, arity: int, *, prefix: str) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("p", label=parent_label, time=0.0, uid=f"{prefix}p")
    for j in range(arity):
        cn = f"{prefix}c{j}"
        G.add_node(cn, label=child_label, time=1.0 + j, uid=cn)
        G.add_edge("p", cn)
    return G


def uid_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:
    return {(G.nodes[u].get("uid", u), G.nodes[v].get("uid", v)) for u, v in G.edges}


def test_star_burst_contracts_all_candidate_children_at_once() -> None:
    G = star_graph("A", "B", 4, prefix="s")
    coarsener = EdgeBPEWithStarMergesCoarsener(num_merges=1, min_pair_count=2).fit([G])

    token = edge_star_token(0, 4)
    assert token in coarsener.encoder_.vocab.entries
    entry = coarsener.encoder_.vocab.entries[token]
    assert entry.parent == (-1, 0, 0, 0, 0)
    assert entry.label == (base_token("A"),) + (base_token("B"),) * 4
    assert entry.attach == (0, 0, 0, 0)

    H = coarsener.transform(G)
    # All five raw nodes collapse into a single arity-4 star node.
    assert H.number_of_nodes() == 1
    assert [data["label"] for _, data in H.nodes(data=True)] == [token]

    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(G)


def test_mixed_arities_under_different_parents_create_per_arity_tokens() -> None:
    G = nx.DiGraph()
    G.add_node("r", label="R", time=0.0, uid="r")
    for pname, k in [("P1", 3), ("P2", 2)]:
        G.add_node(pname, label="P", time=1.0, uid=pname)
        G.add_edge("r", pname)
        for j in range(k):
            cn = f"{pname}_c{j}"
            G.add_node(cn, label="C", time=2.0 + j, uid=cn)
            G.add_edge(pname, cn)

    coarsener = EdgeBPEWithStarMergesCoarsener(num_merges=1, min_pair_count=2).fit([G])

    assert edge_star_token(0, 2) in coarsener.encoder_.vocab.entries
    assert edge_star_token(0, 3) in coarsener.encoder_.vocab.entries
    assert coarsener.history_[0]["arities"] == (2, 3)
    assert coarsener.history_[0]["count"] == 5

    H = coarsener.transform(G)
    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(G)


def test_single_child_merge_uses_edge_bpe_token() -> None:
    X = [path_graph(["A", "B", "C"], prefix="a"), path_graph(["A", "B", "D"], prefix="b")]
    coarsener = EdgeBPEWithStarMergesCoarsener(num_merges=1, min_pair_count=2).fit(X)

    token = edge_bpe_token(0)
    assert token in coarsener.encoder_.vocab.entries
    entry = coarsener.encoder_.vocab.entries[token]
    assert entry.parent == (-1, 0)
    assert entry.label == (base_token("A"), base_token("B"))
    assert entry.attach == (0,)

    H = coarsener.transform(X[0])
    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(X[0])


def test_multilevel_star_merge_reattaches_grandchildren_and_roundtrips() -> None:
    G = nx.DiGraph()
    G.add_node("A", label="A", time=0.0, uid="A")
    for i in range(3):
        b = f"B{i}"
        G.add_node(b, label="B", time=1.0 + i, uid=b)
        G.add_edge("A", b)
        for j, gl in enumerate(["X", "Y"]):
            g = f"{b}_{gl}"
            G.add_node(g, label=gl, time=10.0 + j, uid=g)
            G.add_edge(b, g)

    coarsener = EdgeBPEWithStarMergesCoarsener(num_merges=5, min_pair_count=2).fit([G])
    H = coarsener.transform(G)
    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(G)


def test_fit_is_deterministic() -> None:
    G = star_graph("A", "B", 4, prefix="s")
    first = EdgeBPEWithStarMergesCoarsener(num_merges=3, min_pair_count=2).fit([G])
    second = EdgeBPEWithStarMergesCoarsener(num_merges=3, min_pair_count=2).fit([G])
    assert [h["tokens"] for h in first.history_] == [h["tokens"] for h in second.history_]


def test_transform_leaves_arity_absent_from_vocabulary_uncontracted() -> None:
    # Fit on arity-2 stars only, then transform a fresh arity-3 star. The arity-3
    # token was never learned, so those children are left uncontracted, matching
    # the star coarsener's fixed-vocabulary behavior.
    train = [
        star_graph("A", "B", 2, prefix="t1"),
        star_graph("A", "B", 2, prefix="t2"),
    ]
    coarsener = EdgeBPEWithStarMergesCoarsener(num_merges=1, min_pair_count=2).fit(train)
    assert edge_star_token(0, 2) in coarsener.encoder_.vocab.entries
    assert edge_star_token(0, 3) not in coarsener.encoder_.vocab.entries

    fresh = star_graph("A", "B", 3, prefix="u")
    H = coarsener.transform(fresh)
    # Parent plus three uncontracted children remain (no arity-3 token exists).
    assert H.number_of_nodes() == 4
    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(fresh)
