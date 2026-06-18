"""Small reproducible benchmark for the optional edge-BPE Numba backend.

Examples
--------
python benchmarks/benchmark_edge_bpe_numba.py path --nodes 100000
python benchmarks/benchmark_edge_bpe_numba.py star --nodes 100000

To force a true compilation-cold run, point ``NUMBA_CACHE_DIR`` at a new empty
folder before invoking the script.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import networkx as nx

from tree_coarsening import EdgeBPECoarsener


def make_path(n_nodes: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (
            node,
            {
                "label": "A" if node % 2 == 0 else "B",
                "time": float(node),
                "uid": node,
            },
        )
        for node in range(n_nodes)
    )
    graph.add_edges_from((node - 1, node) for node in range(1, n_nodes))
    return graph


def make_star(n_nodes: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, label="A", time=0.0, uid=0)
    graph.add_nodes_from(
        (
            node,
            {"label": "B", "time": float(node), "uid": node},
        )
        for node in range(1, n_nodes)
    )
    graph.add_edges_from((0, node) for node in range(1, n_nodes))
    return graph


def timed_fit(
    graph: nx.DiGraph,
    *,
    backend: str,
    num_merges: int,
    min_pair_count: int,
    validate_inputs: bool,
) -> tuple[float, int]:
    model = EdgeBPECoarsener(
        backend=backend,
        num_merges=num_merges,
        min_pair_count=min_pair_count,
        validate_inputs=validate_inputs,
        model_id="benchmark",
    )
    start = perf_counter()
    model.fit([graph])
    return perf_counter() - start, len(model.history_)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", choices=("path", "star"))
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--merges", type=int, default=20)
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    graph = (make_path if args.shape == "path" else make_star)(args.nodes)
    common = {
        "num_merges": args.merges,
        "min_pair_count": args.min_pair_count,
        "validate_inputs": args.validate,
    }

    python_time, python_rules = timed_fit(graph, backend="python", **common)
    first_time, first_rules = timed_fit(graph, backend="numba", **common)
    warm_time, warm_rules = timed_fit(graph, backend="numba", **common)

    print(f"shape={args.shape} nodes={args.nodes:,} merges={args.merges}")
    print(f"python:      {python_time:9.4f}s  rules={python_rules}")
    print(f"numba first: {first_time:9.4f}s  rules={first_rules}")
    print(f"numba warm:  {warm_time:9.4f}s  rules={warm_rules}")


if __name__ == "__main__":
    main()
