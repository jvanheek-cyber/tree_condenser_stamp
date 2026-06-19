"""Run the deterministic round-trip suite without invoking pytest.

Usage
-----
python tests/roundtrip_suite/run_roundtrip_checks.py
"""

from __future__ import annotations

# Permit both ``pytest`` collection and direct ``python path/to/script.py`` use.
if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.roundtrip_suite.test_schema_and_backend_invariants import main as schema_main
from tests.roundtrip_suite.test_simulation_reproducibility import main as simulation_main
from tests.roundtrip_suite.test_single_stage_roundtrips import main as single_main
from tests.roundtrip_suite.test_two_stage_roundtrips import main as two_stage_main


def main() -> None:
    simulation_main()
    single_main()
    two_stage_main()
    schema_main()
    print("all deterministic round-trip checks passed")


if __name__ == "__main__":
    main()
