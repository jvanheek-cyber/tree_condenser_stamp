# Deterministic round-trip suite

This folder contains the high-level deterministic checks requested for the
coarsening pipeline.  The Python files are ordinary `pytest` tests and can also
be run as a single direct command:

```bash
pytest -q tests/roundtrip_suite
python tests/roundtrip_suite/run_roundtrip_checks.py
```

The visual notebook imports the same constructors and seed constants from
`common.py`:

```text
coarsening_roundtrip_visual_checks.ipynb
```

It repeats the central assertions while plotting:

- seeded simulated raw trees;
- one Star round trip;
- one edge-BPE round trip;
- one named-component round trip;
- Star → BPE and reverse-stage decoding;
- BPE → Star and reverse-stage decoding.

The fixed seeds are:

```text
STAR_SEED     = 17041
BPE_SEED      = 29117
NAMED_SEED    = 31337
PIPELINE_SEED = 44021
```
