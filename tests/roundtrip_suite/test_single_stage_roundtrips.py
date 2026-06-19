"""Single-stage Star, edge-BPE, and named-component round-trip tests."""

from __future__ import annotations

# Permit both ``pytest`` collection and direct ``python path/to/script.py`` use.
if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

from tree_coarsening import (
    EdgeBPECoarsener,
    NamedVertexCoarsener,
    StarCoarsener,
)

from tests.roundtrip_suite.common import (
    assert_encoded_invariants,
    assert_raw_roundtrip,
    make_bpe_graphs,
    make_named_graph,
    make_star_graphs,
)


def test_star_single_stage_roundtrip() -> None:
    graphs = make_star_graphs()
    model = StarCoarsener(d=4, m=2, contract_d=3).fit(graphs)
    encoded = model.transform(graphs)

    assert len(model.encoder_.vocab.entries) > 0
    assert any(H.number_of_nodes() < G.number_of_nodes() for G, H in zip(graphs, encoded))
    for original, coarse in zip(graphs, encoded):
        assert_encoded_invariants(original, coarse)
        assert_raw_roundtrip(original, model.inverse_transform(coarse))


def test_edge_bpe_single_stage_roundtrip() -> None:
    graphs = make_bpe_graphs()
    model = EdgeBPECoarsener(num_merges=4, min_pair_count=4).fit(graphs)
    encoded = model.transform(graphs)

    assert len(model.history_) == 4
    assert all(record["count"] >= record["actual_events"] >= 1 for record in model.history_)
    assert any(H.number_of_nodes() < G.number_of_nodes() for G, H in zip(graphs, encoded))
    for original, coarse in zip(graphs, encoded):
        assert_encoded_invariants(original, coarse)
        assert_raw_roundtrip(original, model.inverse_transform(coarse))


def test_named_component_single_stage_roundtrip() -> None:
    graph = make_named_graph()
    model = NamedVertexCoarsener(
        labels={"A", "B"},
        component_policy="all",
    ).fit(graph)
    encoded = model.transform(graph)

    assert encoded.number_of_nodes() < graph.number_of_nodes()
    assert_encoded_invariants(graph, encoded)
    assert_raw_roundtrip(graph, model.inverse_transform(encoded))


def main() -> None:
    test_star_single_stage_roundtrip()
    test_edge_bpe_single_stage_roundtrip()
    test_named_component_single_stage_roundtrip()
    print("single-stage round-trip checks passed")


if __name__ == "__main__":
    main()
