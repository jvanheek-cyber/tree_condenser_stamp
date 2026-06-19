"""Compare Python, first-call Numba, and warmed Numba edge-BPE fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        (node, {"label": "B", "time": float(node), "uid": node})
        for node in range(1, n_nodes)
    )
    graph.add_edges_from((0, node) for node in range(1, n_nodes))
    return graph


def timed_fit(
    graph: nx.DiGraph,
    backend: str,
    merges: int,
) -> tuple[float, EdgeBPECoarsener]:
    model = EdgeBPECoarsener(
        backend=backend,
        num_merges=merges,
        min_pair_count=2,
        validate_inputs=False,
        model_id="benchmark",
    )
    start = perf_counter()
    model.fit([graph])
    return perf_counter() - start, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", choices=("path", "star"), nargs="?", default="path")
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--merges", type=int, default=20)
    args = parser.parse_args()

    graph = (make_path if args.shape == "path" else make_star)(args.nodes)
    python_time, python_model = timed_fit(graph, "python", args.merges)
    first_time, first_model = timed_fit(graph, "numba", args.merges)
    warm_time, warm_model = timed_fit(graph, "numba", args.merges)

    assert first_model.history_ == python_model.history_
    assert warm_model.history_ == python_model.history_
    print(f"shape={args.shape} nodes={args.nodes:,} merges={args.merges}")
    print(f"python:      {python_time:9.4f}s  rules={len(python_model.history_)}")
    print(f"numba first: {first_time:9.4f}s  rules={len(first_model.history_)}")
    print(f"numba warm:  {warm_time:9.4f}s  rules={len(warm_model.history_)}")


if __name__ == "__main__":
    main()
