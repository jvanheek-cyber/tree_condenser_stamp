"""Deterministic simulator tests used by pytest and the visual notebook."""

from __future__ import annotations

# Permit both ``pytest`` collection and direct ``python path/to/script.py`` use.
if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

from tree_coarsening.validation import validate_raw_tree

from tests.roundtrip_suite.common import (
    BPE_SEED,
    NAMED_SEED,
    STAR_SEED,
    assert_reproducible,
    make_bpe_graphs,
    make_named_graph,
    make_star_graphs,
    raw_signature,
)


def test_starburst_simulation_is_seeded_and_valid() -> None:
    first = make_star_graphs()
    second = make_star_graphs()
    assert_reproducible(first, second)
    for graph in first:
        validate_raw_tree(graph, require_uid=True)


def test_edge_bpe_simulation_is_seeded_and_valid() -> None:
    first = make_bpe_graphs()
    second = make_bpe_graphs()
    assert_reproducible(first, second)
    for graph in first:
        validate_raw_tree(graph, require_uid=True)


def test_named_component_simulation_is_seeded_and_valid() -> None:
    first = make_named_graph()
    second = make_named_graph()
    assert raw_signature(first) == raw_signature(second)
    validate_raw_tree(first, require_uid=True)


def test_seed_constants_are_distinct() -> None:
    assert len({STAR_SEED, BPE_SEED, NAMED_SEED}) == 3


def main() -> None:
    test_starburst_simulation_is_seeded_and_valid()
    test_edge_bpe_simulation_is_seeded_and_valid()
    test_named_component_simulation_is_seeded_and_valid()
    test_seed_constants_are_distinct()
    print("simulation reproducibility checks passed")


if __name__ == "__main__":
    main()
