"""Shared deterministic fixtures and assertions for round-trip tests.

The notebook in this directory imports these same seed values and constructors,
so the visual checks exercise exactly the same examples as the automated tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import networkx as nx

from tree_coarsening import validate_coarsenable_tree
from tree_coarsening.utils import (
    make_edge_bpe_dataset,
    make_named_component_tree,
    make_starburst_dataset,
)

STAR_SEED = 17_041
BPE_SEED = 29_117
NAMED_SEED = 31_337
PIPELINE_SEED = 44_021
VISUAL_LAYOUT_SEED = 202_503


def make_star_graphs() -> list[nx.DiGraph]:
    """Star-heavy corpus with stable labels, UIDs, topology, and times."""

    return make_starburst_dataset(
        n_graphs=4,
        max_nodes=24,
        n_bursts=3,
        burst_size_range=(4, 6),
        parent_label="P",
        child_label="S",
        tail_label="T",
        tail_probability=0.35,
        seed=STAR_SEED,
    )


def make_bpe_graphs() -> list[nx.DiGraph]:
    """Repeated edge-motif corpus that reliably learns several BPE rules."""

    return make_edge_bpe_dataset(
        n_graphs=3,
        n_repeats=7,
        motif_labels=("A", "B", "C", "D"),
        anchor_labels=("X", "Y", "Z", "W"),
        seed=BPE_SEED,
    )


def make_named_graph() -> nx.DiGraph:
    """Tree with deterministic separated components selected by labels A/B."""

    return make_named_component_tree(
        component_sizes=(6, 4),
        selected_labels=("A", "B"),
        include_singleton=True,
        seed=NAMED_SEED,
        uid_prefix="named_test",
    )


def make_bpe_then_star_graphs() -> list[nx.DiGraph]:
    """Repeated A→B branches: BPE merges each edge, then Star merges siblings."""

    graphs: list[nx.DiGraph] = []
    for graph_i in range(2):
        graph = nx.DiGraph()
        prefix = f"pipeline_{PIPELINE_SEED}_{graph_i}_"
        graph.add_node(0, label="ROOT", time=0.0, uid=f"{prefix}root")
        next_node = 1
        for branch in range(6):
            parent = next_node
            child = next_node + 1
            next_node += 2
            graph.add_node(
                parent,
                label="A",
                time=1.0 + branch / 100,
                uid=f"{prefix}a_{branch}",
            )
            graph.add_node(
                child,
                label="B",
                time=2.0 + branch / 100,
                uid=f"{prefix}b_{branch}",
            )
            graph.add_edge(0, parent)
            graph.add_edge(parent, child)
        graphs.append(graph)
    return graphs


def uid_node_records(graph: nx.DiGraph) -> dict[Any, tuple[Any, float]]:
    return {
        data.get("uid", node): (data["label"], float(data["time"]))
        for node, data in graph.nodes(data=True)
    }


def uid_edges(graph: nx.DiGraph) -> set[tuple[Any, Any]]:
    return {
        (
            graph.nodes[parent].get("uid", parent),
            graph.nodes[child].get("uid", child),
        )
        for parent, child in graph.edges
    }


def raw_signature(graph: nx.DiGraph) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    nodes = tuple(
        sorted(
            (
                repr(uid),
                repr(label),
                time,
            )
            for uid, (label, time) in uid_node_records(graph).items()
        )
    )
    edges = tuple(sorted((repr(parent), repr(child)) for parent, child in uid_edges(graph)))
    return nodes, edges


def encoded_signature(graph: nx.DiGraph) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Node-key-independent signature for one encoded stage."""

    node_records = []
    group_for_node: dict[Any, tuple[Any, ...]] = {}
    for node, data in graph.nodes(data=True):
        uids = tuple(data["super_uids"])
        group_for_node[node] = uids
        node_records.append(
            (
                tuple(map(repr, uids)),
                repr(data["label"]),
                repr(data["type"]),
                int(data["size"]),
                float(data["time"]),
            )
        )
    edge_records = [
        (
            tuple(map(repr, group_for_node[parent])),
            tuple(map(repr, group_for_node[child])),
            tuple(edge_data["attach_map"]),
        )
        for parent, child, edge_data in graph.edges(data=True)
    ]
    return tuple(sorted(node_records)), tuple(sorted(edge_records))


def assert_raw_roundtrip(original: nx.DiGraph, recovered: nx.DiGraph) -> None:
    """Assert exact UID topology, labels, and times after full stage reversal."""

    assert uid_edges(recovered) == uid_edges(original)
    original_nodes = uid_node_records(original)
    recovered_nodes = uid_node_records(recovered)
    assert recovered_nodes.keys() == original_nodes.keys()
    for uid, (label, time) in original_nodes.items():
        recovered_label, recovered_time = recovered_nodes[uid]
        assert recovered_label == label
        assert math.isclose(recovered_time, time, rel_tol=0.0, abs_tol=1e-12)


def assert_encoded_stage_equal(expected: nx.DiGraph, actual: nx.DiGraph) -> None:
    """Assert that stage-local decoding recovered the previous encoded stage."""

    assert encoded_signature(actual) == encoded_signature(expected)


def assert_encoded_invariants(
    raw_graph: nx.DiGraph,
    encoded_graph: nx.DiGraph,
) -> None:
    """Check common schema, UID partition, additive sizes, max times, and maps."""

    validate_coarsenable_tree(encoded_graph)
    assert nx.is_arborescence(encoded_graph)

    raw_by_uid = {
        data["uid"]: data
        for _, data in raw_graph.nodes(data=True)
    }
    seen: list[Any] = []
    for _, data in encoded_graph.nodes(data=True):
        uids = tuple(data["super_uids"])
        assert data["size"] == len(uids)
        assert data["size"] >= 1
        assert data["type"] is not None
        assert uids
        expected_time = max(float(raw_by_uid[uid]["time"]) for uid in uids)
        assert math.isclose(float(data["time"]), expected_time, rel_tol=0.0, abs_tol=1e-12)
        seen.extend(uids)

    assert len(seen) == raw_graph.number_of_nodes()
    assert set(seen) == set(raw_by_uid)
    assert len(seen) == len(set(seen))
    assert sum(data["size"] for _, data in encoded_graph.nodes(data=True)) == raw_graph.number_of_nodes()

    for _, _, data in encoded_graph.edges(data=True):
        attach_map = data["attach_map"]
        assert isinstance(attach_map, tuple)
        assert attach_map
        assert all(isinstance(site, int) and site >= 0 for site in attach_map)


def assert_reproducible(
    first: Iterable[nx.DiGraph],
    second: Iterable[nx.DiGraph],
) -> None:
    assert [raw_signature(graph) for graph in first] == [
        raw_signature(graph) for graph in second
    ]


def stage_summary(name: str, raw: nx.DiGraph, encoded: nx.DiGraph) -> dict[str, Any]:
    return {
        "stage": name,
        "raw_nodes": raw.number_of_nodes(),
        "encoded_nodes": encoded.number_of_nodes(),
        "compression_ratio": encoded.number_of_nodes() / raw.number_of_nodes(),
        "encoded_edges": encoded.number_of_edges(),
        "represented_size": sum(data["size"] for _, data in encoded.nodes(data=True)),
    }


def token_text(token: Any) -> str:
    if isinstance(token, tuple) and token:
        if token[0] == "base" and len(token) == 2:
            return str(token[1])
        if token[0] == "star" and len(token) == 4:
            return f"star({token[1]}→{token[2]}×{token[3]})"
        if token[0] == "edge_bpe" and len(token) == 2:
            return f"BPE#{token[1]}"
        if token[0] == "named_component":
            return f"named[{token[2]}]"
    return str(token)
