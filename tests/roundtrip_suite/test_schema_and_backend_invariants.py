"""Additional deterministic invariants worth keeping around the base suite."""

from __future__ import annotations

# Permit both ``pytest`` collection and direct ``python path/to/script.py`` use.
if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from tree_coarsening import EdgeBPECoarsener, StarCoarsener

from tests.roundtrip_suite.common import (
    assert_encoded_invariants,
    make_bpe_graphs,
    make_star_graphs,
)


def test_numba_and_python_bpe_learn_the_same_history() -> None:
    pytest.importorskip("numba")
    graphs = make_bpe_graphs()
    kwargs = dict(num_merges=4, min_pair_count=4, model_id="roundtrip-backend-parity")
    python_model = EdgeBPECoarsener(backend="python", **kwargs).fit(graphs)
    numba_model = EdgeBPECoarsener(backend="numba", **kwargs).fit(graphs)
    assert numba_model.history_ == python_model.history_


def test_each_stage_preserves_total_size_and_max_time() -> None:
    raw = make_star_graphs()
    star = StarCoarsener(d=4, m=2, contract_d=3).fit(raw)
    intermediate = star.transform(raw)
    bpe = EdgeBPECoarsener(num_merges=8, min_pair_count=2).fit(intermediate)
    final = bpe.transform(intermediate)

    for original, first, second in zip(raw, intermediate, final):
        assert_encoded_invariants(original, first)
        assert_encoded_invariants(original, second)


def main() -> None:
    test_each_stage_preserves_total_size_and_max_time()
    try:
        test_numba_and_python_bpe_learn_the_same_history()
    except pytest.skip.Exception:
        print("Numba not installed; backend parity check skipped")
    print("schema/backend invariant checks passed")


if __name__ == "__main__":
    main()
