# Experimental Numba backend for edge BPE

`EdgeBPECoarsener` accepts:

```python
EdgeBPECoarsener(..., backend="python")  # default
EdgeBPECoarsener(..., backend="numba")   # optional experiment
```

Install the optional dependency with:

```bash
pip install -e ".[numba]"
```

## Scope

The experimental backend accelerates fitting only. It compiles:

- a flattened forest representation based on NumPy arrays;
- the incremental edge-pair occurrence index;
- local pair-count updates during in-place edge contraction.

The following remain Python:

- validation and NetworkX preprocessing;
- deterministic candidate sorting;
- vocabulary and rule construction;
- `transform` and decoding.

The fitted rules, raw matching-edge scores, deterministic contraction order,
and encoder/decoder artifacts are intended to be identical to the Python
backend.

## Why a separate implementation is needed

The Python backend indexes occurrences as a dictionary from an edge-key triple
to a set of child indices. That representation is convenient in Python but is
not a good Numba boundary. The compiled backend therefore uses:

- fixed NumPy arrays for parent, child/sibling links, label, time, attachment,
  and liveness;
- a compiled dictionary from edge keys to integer bucket IDs;
- linked occurrence lists represented by integer arrays;
- dynamic bucket metadata represented by typed integer lists.

This is materially more code than adding `@njit` to the existing methods.

## Benchmarking

Run:

```bash
python benchmarks/benchmark_edge_bpe_numba.py path --nodes 100000
python benchmarks/benchmark_edge_bpe_numba.py star --nodes 100000
```

The script reports the Python fit, the first Numba fit, and a second warmed
Numba fit. The first fit includes compilation unless a usable disk cache is
already present. To force a cold compilation, set `NUMBA_CACHE_DIR` to a new
empty directory before running the script.

## Status

This backend is experimental and is not the default. It is most plausible for
large fits or repeated fits in one long-running process. For small jobs and
interactive tests, JIT startup can cost more than the compiled kernel saves.
