from __future__ import annotations

import networkx as nx

from tree_coarsening import (
    StarCoarsener,
    normalize_coarsenable_tree,
    validate_coarsenable_tree,
)


REQUIRED_FIELDS = {
    "label",
    "type",
    "size",
    "time",
    "super_label",
    "super_uids",
}


def raw_star() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="P", time=1.0, uid="p")
    for node, time in [(1, 2.0), (2, 4.0), (3, 3.0)]:
        graph.add_node(node, label="C", time=time, uid=f"c{node}")
        graph.add_edge(0, node)
    return graph


def test_raw_tree_normalizes_to_common_fitting_contract() -> None:
    graph = normalize_coarsenable_tree(raw_star())
    validate_coarsenable_tree(graph)
    for node, data in graph.nodes(data=True):
        assert REQUIRED_FIELDS <= set(data)
        assert data["size"] == 1
        assert data["super_uids"] == (data["uid"],)
    assert all(data["attach_map"] == (0,) for *_edge, data in graph.edges(data=True))


def test_transform_accepts_one_or_list_and_emits_max_time_and_size() -> None:
    graph = raw_star()
    coarsener = StarCoarsener(d=3, m=1).fit(graph)

    single = coarsener.transform(graph)
    batch = coarsener.transform([graph, graph])
    assert isinstance(single, nx.DiGraph)
    assert isinstance(batch, list) and len(batch) == 2

    decoded_single = coarsener.inverse_transform(single)
    decoded_batch = coarsener.inverse_transform(batch)
    assert isinstance(decoded_single, nx.DiGraph)
    assert isinstance(decoded_batch, list) and len(decoded_batch) == 2

    validate_coarsenable_tree(single)
    for _, data in single.nodes(data=True):
        assert set(data) == REQUIRED_FIELDS
        assert data["size"] == len(data["super_uids"])
        assert data["type"] is not None

    star_node = next(node for node, data in single.nodes(data=True) if data["size"] == 3)
    assert single.nodes[star_node]["time"] == 4.0
