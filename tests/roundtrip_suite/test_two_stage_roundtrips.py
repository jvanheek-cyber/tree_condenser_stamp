"""Two-stage pipeline tests with explicit reverse-stage decoding."""

from __future__ import annotations

# Permit both ``pytest`` collection and direct ``python path/to/script.py`` use.
if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

from tree_coarsening import EdgeBPECoarsener, StarCoarsener

from tests.roundtrip_suite.common import (
    assert_encoded_invariants,
    assert_encoded_stage_equal,
    assert_raw_roundtrip,
    make_bpe_then_star_graphs,
    make_star_graphs,
)


def test_star_then_bpe_roundtrip() -> None:
    raw = make_star_graphs()
    star = StarCoarsener(d=4, m=2, contract_d=3).fit(raw)
    star_stage = star.transform(raw)

    bpe = EdgeBPECoarsener(num_merges=10, min_pair_count=2).fit(star_stage)
    final_stage = bpe.transform(star_stage)

    assert len(bpe.history_) > 0
    for original, intermediate, final in zip(raw, star_stage, final_stage):
        assert_encoded_invariants(original, intermediate)
        assert_encoded_invariants(original, final)
        recovered_intermediate = bpe.inverse_transform(final)
        assert_encoded_stage_equal(intermediate, recovered_intermediate)
        recovered_raw = star.inverse_transform(recovered_intermediate)
        assert_raw_roundtrip(original, recovered_raw)


def test_bpe_then_star_roundtrip() -> None:
    raw = make_bpe_then_star_graphs()
    bpe = EdgeBPECoarsener(num_merges=1, min_pair_count=2).fit(raw)
    bpe_stage = bpe.transform(raw)

    star = StarCoarsener(d=4, m=1, contract_d=4).fit(bpe_stage)
    final_stage = star.transform(bpe_stage)

    assert len(bpe.history_) == 1
    assert any(final.number_of_nodes() < intermediate.number_of_nodes() for intermediate, final in zip(bpe_stage, final_stage))
    for original, intermediate, final in zip(raw, bpe_stage, final_stage):
        assert_encoded_invariants(original, intermediate)
        assert_encoded_invariants(original, final)
        recovered_intermediate = star.inverse_transform(final)
        assert_encoded_stage_equal(intermediate, recovered_intermediate)
        recovered_raw = bpe.inverse_transform(recovered_intermediate)
        assert_raw_roundtrip(original, recovered_raw)


def main() -> None:
    test_star_then_bpe_roundtrip()
    test_bpe_then_star_roundtrip()
    print("two-stage round-trip checks passed")


if __name__ == "__main__":
    main()
