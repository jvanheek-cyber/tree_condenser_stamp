from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import NamedVertexCoarsener, base_token


def make_tree() -> nx.DiGraph:
    """Two A->B regions plus unselected separators and boundary children."""

    G = nx.DiGraph()
    nodes = {
        0: ("R", "r"),
        1: ("A", "a1"),
        2: ("B", "b1"),
        3: ("X", "x"),
        4: ("A", "a2"),
        5: ("B", "b2"),
        6: ("B", "b3"),
        7: ("Y", "y1"),
        8: ("Y", "y2"),
    }
    for node, (label, uid) in nodes.items():
        G.add_node(node, label=label, time=float(node), uid=uid)
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 7),
            (0, 3),
            (3, 4),
            (4, 5),
            (4, 6),
            (6, 8),
        ]
    )
    return G


def uid_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:
    return {
        (G.nodes[u].get("uid", u), G.nodes[v].get("uid", v))
        for u, v in G.edges
    }


def uid_label_map(G: nx.DiGraph) -> dict[str, str]:
    return {data.get("uid", node): data["label"] for node, data in G.nodes(data=True)}


def learned_nodes(H: nx.DiGraph) -> list[int]:
    return [node for node, data in H.nodes(data=True) if data["label"][0] == "named_component"]


def test_uid_selector_contracts_all_maximal_components_and_roundtrips() -> None:
    G = make_tree()
    coarsener = NamedVertexCoarsener(uids={"a1", "b1", "a2", "b2", "b3"}).fit([G])

    # No graph-dependent recipe is learned during fit.
    assert len(coarsener.encoder_.vocab.entries) == 0

    H = coarsener.transform(G)
    nodes = learned_nodes(H)
    assert len(nodes) == 2
    assert H.number_of_nodes() == G.number_of_nodes() - 3
    assert {H.nodes[node]["super_uids"] for node in nodes} == {
        ("a1", "b1"),
        ("a2", "b2", "b3"),
    }

    # The transform-time recipes are immediately visible to the shared decoder.
    assert coarsener.encoder_.vocab is coarsener.decoder_.vocab
    assert len(coarsener.encoder_.vocab.entries) == 2

    decoded = coarsener.decode(H)
    assert uid_edge_set(decoded) == uid_edge_set(G)
    assert uid_label_map(decoded) == uid_label_map(G)


def test_largest_policy_contracts_only_largest_component() -> None:
    G = make_tree()
    coarsener = NamedVertexCoarsener(
        uids={"a1", "b1", "a2", "b2", "b3"},
        component_policy="largest",
    ).fit([G])
    H = coarsener.transform(G)

    nodes = learned_nodes(H)
    assert len(nodes) == 1
    assert H.nodes[nodes[0]]["super_uids"] == ("a2", "b2", "b3")
    assert H.number_of_nodes() == G.number_of_nodes() - 2
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)


def test_label_selector_builds_direct_connected_tree_recipe() -> None:
    G = make_tree()
    coarsener = NamedVertexCoarsener(labels={"A", "B"}).fit([G])
    H = coarsener.transform(G)

    nodes = learned_nodes(H)
    assert len(nodes) == 2
    branch_node = next(node for node in nodes if len(H.nodes[node]["super_uids"]) == 3)
    token = H.nodes[branch_node]["label"]
    entry = coarsener.encoder_.vocab.entries[token]

    assert entry.operation == "component"
    assert entry.parent == (-1, 0, 0)
    assert entry.label == ("A", "B", "B")
    assert entry.attach == (0, 0)
    assert coarsener.encoder_.vocab.root_count(token) == 1
    assert coarsener.encoder_.vocab.site_count(token) == 3
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)


def test_singletons_are_not_wrapped_in_identity_tokens() -> None:
    G = make_tree()
    coarsener = NamedVertexCoarsener(uids={"x"}).fit([G])
    H = coarsener.transform(G)

    assert not learned_nodes(H)
    assert H.number_of_nodes() == G.number_of_nodes()
    assert len(coarsener.encoder_.vocab.entries) == 0
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)


def test_fit_is_not_tied_to_fit_graph_component_shapes() -> None:
    fit_graph = nx.DiGraph()
    fit_graph.add_node(0, label="Z", time=0.0, uid="fit-only")

    G = make_tree()
    coarsener = NamedVertexCoarsener(labels={"A", "B"}).fit([fit_graph])
    H = coarsener.transform(G)
    assert len(learned_nodes(H)) == 2
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)


def test_partial_decode_uses_generic_staged_decoder() -> None:
    G = make_tree()
    coarsener = NamedVertexCoarsener(labels={"A", "B"}).fit([G])
    H = coarsener.transform(G)
    target = learned_nodes(H)[0]

    partial = coarsener.decode(H, target=target, by="node", recursive=False)
    assert uid_edge_set(coarsener.decode(partial)) == uid_edge_set(G)


def test_constructor_requires_exactly_one_nonempty_selector() -> None:
    with pytest.raises(ValueError):
        NamedVertexCoarsener()
    with pytest.raises(ValueError):
        NamedVertexCoarsener(uids={"u"}, labels={"A"})
    with pytest.raises(ValueError):
        NamedVertexCoarsener(uids=set())
    with pytest.raises(ValueError):
        NamedVertexCoarsener(labels=set())
    with pytest.raises(ValueError):
        NamedVertexCoarsener(labels={"A"}, component_policy="first")


def test_entire_tree_can_be_one_named_component() -> None:
    G = make_tree()
    labels = {data["label"] for _, data in G.nodes(data=True)}
    coarsener = NamedVertexCoarsener(labels=labels).fit([G])
    H = coarsener.transform(G)

    assert H.number_of_nodes() == 1
    node = next(iter(H.nodes))
    assert H.nodes[node]["label"][0] == "named_component"
    assert len(H.nodes[node]["super_uids"]) == G.number_of_nodes()
    assert uid_edge_set(coarsener.decode(H)) == uid_edge_set(G)


def test_uid_selection_uses_node_key_fallback_when_uid_is_missing() -> None:
    G = nx.DiGraph()
    G.add_node(10, label="A", time=0.0)
    G.add_node(20, label="B", time=1.0)
    G.add_node(30, label="C", time=2.0)
    G.add_edges_from([(10, 20), (20, 30)])

    coarsener = NamedVertexCoarsener(uids={10, 20}).fit([G])
    H = coarsener.transform(G)
    node = learned_nodes(H)[0]
    assert H.nodes[node]["super_uids"] == (10, 20)
    decoded = coarsener.decode(H)
    assert set(decoded.edges) == {(10, 20), (20, 30)}
