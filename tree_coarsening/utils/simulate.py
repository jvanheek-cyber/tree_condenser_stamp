"""Random NetworkX tree generators for bug-testing coarseners.

These helpers are deliberately outside the core method namespace. They build a
simple Galton-Watson-like rooted tree, then decorate selected vertices with
medium-sized starbursts.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import networkx as nx


def random_galton_watson_tree(
    *,
    max_nodes: int = 40,
    mean_children: float = 1.4,
    labels: Sequence[str] = ("A", "B", "C", "D"),
    seed: int | None = None,
    uid_prefix: str = "n",
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
) -> nx.DiGraph:
    """Generate a small directed rooted tree with node labels and times."""

    if max_nodes < 1:
        raise ValueError("max_nodes must be positive.")
    if not labels:
        raise ValueError("labels must be nonempty.")

    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_node(0, **{label_attr: rng.choice(labels), time_attr: 0.0, uid_attr: f"{uid_prefix}0"})
    frontier = [0]
    next_node = 1

    while frontier and next_node < max_nodes:
        parent = frontier.pop(0)
        probs = [0.35, 0.30, 0.20, 0.10, 0.05]
        if mean_children > 1.8:
            probs = [0.20, 0.25, 0.25, 0.20, 0.10]
        elif mean_children < 1.0:
            probs = [0.55, 0.30, 0.12, 0.03, 0.00]
        k = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        for _ in range(k):
            if next_node >= max_nodes:
                break
            t = G.nodes[parent][time_attr] + 1.0 + rng.random()
            G.add_node(
                next_node,
                **{
                    label_attr: rng.choice(labels),
                    time_attr: t,
                    uid_attr: f"{uid_prefix}{next_node}",
                },
            )
            G.add_edge(parent, next_node)
            frontier.append(next_node)
            next_node += 1

    return G


def add_starbursts(
    G: nx.DiGraph,
    *,
    n_bursts: int | None = None,
    num_bursts: int | None = None,
    burst_size_range: tuple[int, int] = (4, 8),
    arity_range: tuple[int, int] | None = None,
    parent_label: str = "P",
    child_label: str = "S",
    tail_probability: float = 0.25,
    tail_label: str | None = "T",
    seed: int | None = None,
    uid_prefix: str = "s",
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
) -> nx.DiGraph:
    """Return a copy of ``G`` decorated with labeled sibling starbursts.

    ``num_bursts`` and ``arity_range`` are accepted as aliases for the newer
    ``n_bursts`` and ``burst_size_range`` names.
    """

    if num_bursts is not None:
        n_bursts = num_bursts
    if n_bursts is None:
        n_bursts = 3
    if arity_range is not None:
        burst_size_range = arity_range
    if n_bursts < 0:
        raise ValueError("n_bursts must be nonnegative.")
    low, high = burst_size_range
    if low < 1 or high < low:
        raise ValueError("burst_size_range must satisfy 1 <= low <= high.")
    if not (0.0 <= tail_probability <= 1.0):
        raise ValueError("tail_probability must lie in [0, 1].")

    rng = random.Random(seed)
    H = G.copy(as_view=False)
    if not H:
        return H
    parents = rng.sample(list(H.nodes), k=min(n_bursts, len(H)))
    next_node = max((n for n in H.nodes if isinstance(n, int)), default=len(H)) + 1
    serial = 0

    for parent in parents:
        H.nodes[parent][label_attr] = parent_label
        base_time = H.nodes[parent][time_attr]
        size = rng.randint(low, high)
        for _ in range(size):
            child = next_node
            next_node += 1
            serial += 1
            H.add_node(
                child,
                **{
                    label_attr: child_label,
                    time_attr: base_time + 0.1 + 0.01 * serial,
                    uid_attr: f"{uid_prefix}{child}",
                },
            )
            H.add_edge(parent, child)
            if tail_label is not None and rng.random() < tail_probability:
                tail = next_node
                next_node += 1
                serial += 1
                H.add_node(
                    tail,
                    **{
                        label_attr: tail_label,
                        time_attr: base_time + 0.2 + 0.01 * serial,
                        uid_attr: f"{uid_prefix}{tail}",
                    },
                )
                H.add_edge(child, tail)
    return H


def make_starburst_dataset(
    *,
    n_graphs: int = 5,
    max_nodes: int = 35,
    n_bursts: int | None = None,
    n_starbursts: int | None = None,
    num_bursts: int | None = None,
    burst_size_range: tuple[int, int] = (4, 7),
    arity_range: tuple[int, int] | None = None,
    parent_label: str = "P",
    child_label: str = "S",
    tail_label: str | None = "T",
    tail_probability: float = 0.25,
    seed: int | None = None,
    base_seed: int | None = None,
    labels: Sequence[str] = ("A", "B", "C", "D"),
) -> list[nx.DiGraph]:
    """Generate a reproducible list of starburst-decorated raw trees.

    Compatibility aliases:
    ``base_seed`` -> ``seed``;
    ``n_starbursts`` or ``num_bursts`` -> ``n_bursts``;
    ``arity_range`` -> ``burst_size_range``.
    """

    if base_seed is not None:
        seed = base_seed
    if seed is None:
        seed = 0
    if n_starbursts is not None:
        n_bursts = n_starbursts
    if num_bursts is not None:
        n_bursts = num_bursts
    if n_bursts is None:
        n_bursts = 3
    if arity_range is not None:
        burst_size_range = arity_range

    rng = random.Random(seed)
    graphs: list[nx.DiGraph] = []
    for i in range(n_graphs):
        base = random_galton_watson_tree(
            max_nodes=max_nodes,
            seed=rng.randint(0, 10**9),
            uid_prefix=f"g{i}_n",
            labels=labels,
        )
        graph = add_starbursts(
            base,
            n_bursts=n_bursts,
            burst_size_range=burst_size_range,
            parent_label=parent_label,
            child_label=child_label,
            tail_label=tail_label,
            tail_probability=tail_probability,
            seed=rng.randint(0, 10**9),
            uid_prefix=f"g{i}_s",
        )
        graphs.append(graph)
    return graphs


def make_repeated_edge_tree(
    *,
    n_repeats: int = 8,
    motif_labels: Sequence[str] = ("A", "B", "C", "D"),
    anchor_labels: Sequence[str] = ("X", "Y", "Z", "W"),
    seed: int | None = None,
    uid_prefix: str = "bpe",
    root_label: str = "ROOT",
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
) -> nx.DiGraph:
    """Build a small tree containing many copies of one labeled edge motif.

    Each repeated branch has the form

    ``root -> anchor -> motif_labels[0] -> ... -> motif_labels[-1]``.

    Anchor labels cycle in a seeded order so that the repeated edges *inside*
    the motif are substantially more frequent than the edges entering it.  This
    makes the result useful for demonstrating hierarchical edge-only BPE: the
    method can first merge the end of the motif, then merge the resulting token
    with the preceding label, and so on.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    if len(motif_labels) < 2:
        raise ValueError("motif_labels must contain at least two labels.")
    if not anchor_labels:
        raise ValueError("anchor_labels must be nonempty.")
    if not all(isinstance(label, str) for label in motif_labels):
        raise TypeError("every motif label must be a string.")
    if not all(isinstance(label, str) for label in anchor_labels):
        raise TypeError("every anchor label must be a string.")
    if not isinstance(root_label, str):
        raise TypeError("root_label must be a string.")

    rng = random.Random(seed)
    anchor_cycle = list(anchor_labels)
    rng.shuffle(anchor_cycle)

    G = nx.DiGraph()
    next_node = 0
    root = next_node
    next_node += 1
    G.add_node(
        root,
        **{
            label_attr: root_label,
            time_attr: 0.0,
            uid_attr: f"{uid_prefix}_root",
        },
    )

    for repeat in range(n_repeats):
        anchor = next_node
        next_node += 1
        anchor_time = 1.0 + 0.01 * repeat
        G.add_node(
            anchor,
            **{
                label_attr: anchor_cycle[repeat % len(anchor_cycle)],
                time_attr: anchor_time,
                uid_attr: f"{uid_prefix}_anchor_{repeat}",
            },
        )
        G.add_edge(root, anchor)

        parent = anchor
        parent_time = anchor_time
        for depth, label in enumerate(motif_labels):
            node = next_node
            next_node += 1
            node_time = parent_time + 1.0 + 0.001 * depth
            G.add_node(
                node,
                **{
                    label_attr: label,
                    time_attr: node_time,
                    uid_attr: f"{uid_prefix}_motif_{repeat}_{depth}",
                },
            )
            G.add_edge(parent, node)
            parent = node
            parent_time = node_time

    return G


def make_edge_bpe_dataset(
    *,
    n_graphs: int = 3,
    n_repeats: int = 8,
    motif_labels: Sequence[str] = ("A", "B", "C", "D"),
    anchor_labels: Sequence[str] = ("X", "Y", "Z", "W"),
    seed: int | None = None,
) -> list[nx.DiGraph]:
    """Return repeated-motif trees suitable for an edge-BPE syntax example."""

    if n_graphs < 1:
        raise ValueError("n_graphs must be positive.")
    rng = random.Random(seed)
    return [
        make_repeated_edge_tree(
            n_repeats=n_repeats,
            motif_labels=motif_labels,
            anchor_labels=anchor_labels,
            seed=rng.randint(0, 10**9),
            uid_prefix=f"bpe_g{i}",
        )
        for i in range(n_graphs)
    ]


def make_named_component_tree(
    *,
    component_sizes: Sequence[int] = (5, 3),
    selected_labels: Sequence[str] = ("A", "B"),
    include_singleton: bool = True,
    seed: int | None = None,
    uid_prefix: str = "named",
    root_label: str = "ROOT",
    spacer_label: str = "X",
    outside_label: str = "O",
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
) -> nx.DiGraph:
    """Build a tree with several separated, easily named components.

    Each requested component is a small rooted binary tree whose labels are
    drawn from ``selected_labels``.  Nonselected spacer vertices separate the
    components, and each nontrivial component has one nonselected outgoing leaf
    so that examples exercise both incoming and outgoing token boundaries.

    Component UIDs use the predictable form
    ``"{uid_prefix}_component_<component>_<site>"``.  A notebook can therefore
    demonstrate UID selection without relying on NetworkX node keys.
    """

    sizes = tuple(component_sizes)
    if not sizes:
        raise ValueError("component_sizes must be nonempty.")
    if any(size < 1 for size in sizes):
        raise ValueError("every component size must be positive.")
    selected_label_values = tuple(selected_labels)
    if not selected_label_values:
        raise ValueError("selected_labels must be nonempty.")
    if not all(isinstance(label, str) for label in selected_label_values):
        raise TypeError("every selected label must be a string.")
    for name, value in (
        ("root_label", root_label),
        ("spacer_label", spacer_label),
        ("outside_label", outside_label),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        if value in selected_label_values:
            raise ValueError(f"{name} must not be one of selected_labels.")

    rng = random.Random(seed)
    G = nx.DiGraph()
    next_node = 0
    root = next_node
    next_node += 1
    G.add_node(
        root,
        **{
            label_attr: root_label,
            time_attr: 0.0,
            uid_attr: f"{uid_prefix}_root",
        },
    )

    for component_i, size in enumerate(sizes):
        spacer = next_node
        next_node += 1
        spacer_time = 1.0 + component_i
        G.add_node(
            spacer,
            **{
                label_attr: spacer_label,
                time_attr: spacer_time,
                uid_attr: f"{uid_prefix}_spacer_{component_i}",
            },
        )
        G.add_edge(root, spacer)

        component_nodes: list[int] = []
        for site_i in range(size):
            node = next_node
            next_node += 1
            if site_i == 0:
                parent = spacer
            else:
                parent = component_nodes[(site_i - 1) // 2]
            node_time = float(G.nodes[parent][time_attr]) + 1.0 + 0.001 * site_i
            G.add_node(
                node,
                **{
                    label_attr: rng.choice(selected_label_values),
                    time_attr: node_time,
                    uid_attr: f"{uid_prefix}_component_{component_i}_{site_i}",
                },
            )
            G.add_edge(parent, node)
            component_nodes.append(node)

        boundary_parent = component_nodes[-1]
        outside = next_node
        next_node += 1
        G.add_node(
            outside,
            **{
                label_attr: outside_label,
                time_attr: float(G.nodes[boundary_parent][time_attr]) + 1.0,
                uid_attr: f"{uid_prefix}_outside_{component_i}",
            },
        )
        G.add_edge(boundary_parent, outside)

    if include_singleton:
        singleton = next_node
        G.add_node(
            singleton,
            **{
                label_attr: rng.choice(selected_label_values),
                time_attr: 1.0 + len(sizes),
                uid_attr: f"{uid_prefix}_singleton",
            },
        )
        G.add_edge(root, singleton)

    return G
