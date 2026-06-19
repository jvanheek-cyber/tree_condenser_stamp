from __future__ import annotations

import networkx as nx

from tree_coarsening import CompositeType, StarCoarsener, base_token, star_token
from tree_coarsening.provenance import PROVENANCE_KEY


EXPECTED_ENCODED_NODE_FIELDS = {
    "label", "type", "size", "time", "super_label", "super_uids"
}
from tree_coarsening.utils import make_starburst_dataset


def uid_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (G.nodes[u].get("uid", u), G.nodes[v].get("uid", v))
        for u, v in G.edges
    }


def uid_label_map(G: nx.DiGraph) -> dict[str, str]:
    return {data.get("uid", node): data["label"] for node, data in G.nodes(data=True)}


def test_star_schema_is_minimal_and_integer_keyed() -> None:
    X = make_starburst_dataset(
        n_graphs=2,
        seed=4,
        max_nodes=10,
        n_bursts=2,
        burst_size_range=(4, 4),
        tail_probability=0.0,
        parent_label="P",
        child_label="S",
    )
    coarsener = StarCoarsener(d=3, m=1).fit(X)
    H = coarsener.transform(X[0])

    assert list(H.nodes) == list(range(H.number_of_nodes()))
    assert PROVENANCE_KEY in H.graph

    for _, data in H.nodes(data=True):
        assert set(data) == EXPECTED_ENCODED_NODE_FIELDS
        assert "uid" not in data
        assert isinstance(data["time"], float)
        assert isinstance(data["super_uids"], tuple)
        assert data["size"] == len(data["super_uids"])
        assert data["label"] in coarsener.encoder_.vocab
        if isinstance(data["type"], CompositeType):
            assert data["type"].label == data["label"]
        else:
            assert data["type"] == base_token(data["label"])

    assert any(data["label"] == star_token("P", "S", 4) for _, data in H.nodes(data=True))
    assert any(data["label"] == "P" for _, data in H.nodes(data=True))


def test_full_roundtrip_recovers_labels_times_and_edges_by_uid() -> None:
    X = make_starburst_dataset(
        n_graphs=2,
        seed=7,
        max_nodes=14,
        n_bursts=3,
        burst_size_range=(4, 5),
        tail_probability=0.3,
        parent_label="P",
        child_label="S",
        tail_label="T",
    )
    coarsener = StarCoarsener(d=3, m=1).fit(X)
    H = coarsener.transform(X[0])
    G2 = coarsener.decode(H)

    raw = X[0]
    assert set(uid_label_map(raw)) == set(G2.nodes)
    assert uid_label_map(raw) == uid_label_map(G2)
    assert uid_edge_set(raw) == set(G2.edges)
    for uid in G2.nodes:
        original = next(data for _, data in raw.nodes(data=True) if data["uid"] == uid)
        assert G2.nodes[uid]["time"] == original["time"]
        assert G2.nodes[uid]["uid"] == uid


def test_partial_decode_by_node_and_by_label_roundtrip() -> None:
    X = make_starburst_dataset(
        n_graphs=2,
        seed=10,
        max_nodes=12,
        n_bursts=2,
        burst_size_range=(4, 4),
        tail_probability=0.25,
        parent_label="P",
        child_label="S",
    )
    coarsener = StarCoarsener(d=3, m=1).fit(X)
    H = coarsener.transform(X[0])
    star_nodes = [n for n, data in H.nodes(data=True) if data["label"] == star_token("P", "S", 4)]
    assert star_nodes

    H_node = coarsener.decode(H, target=star_nodes[0], by="node", recursive=False)
    assert all(set(data) == EXPECTED_ENCODED_NODE_FIELDS for _, data in H_node.nodes(data=True))
    assert uid_edge_set(X[0]) == set(coarsener.decode(H_node).edges)

    H_label = coarsener.decode(H, target=star_token("P", "S", 4), by="label", recursive=True)
    assert uid_edge_set(X[0]) == set(coarsener.decode(H_label).edges)


def test_contract_d_can_be_smaller_than_witness_d() -> None:
    G = nx.DiGraph()
    G.add_node(0, label="R", time=0.0, uid="r")
    G.add_node(1, label="P", time=1.0, uid="p5")
    G.add_node(2, label="P", time=1.0, uid="p3")
    G.add_edge(0, 1)
    G.add_edge(0, 2)

    next_node = 3
    for parent, count in [(1, 5), (2, 3)]:
        for _ in range(count):
            G.add_node(next_node, label="S", time=float(next_node), uid=f"s{next_node}")
            G.add_edge(parent, next_node)
            next_node += 1

    coarsener = StarCoarsener(d=5, m=1, contract_d=3).fit([G])
    H = coarsener.transform(G)
    labels = [data["label"] for _, data in H.nodes(data=True)]
    assert star_token("P", "S", 5) in labels
    assert star_token("P", "S", 3) in labels
    assert set(uid_edge_set(G)) == set(coarsener.decode(H).edges)
