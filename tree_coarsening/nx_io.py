"""NetworkX schema helpers for encoded tree objects."""

from __future__ import annotations

from typing import Any, Hashable

import networkx as nx

from .vocabulary import AttachMap, normalize_attach_map


def edge_attach_attrs(
    attach_map: Any,
    *,
    attach_attr: str = "attach_map",
    include_scalar_shorthand: bool = False,
) -> dict[str, Any]:
    """Return edge attributes containing normalized tuple-valued gluing data.

    The current public schema stores only ``attach_map``. The scalar
    ``attach_index`` shorthand can still be produced at explicit user
    boundaries by setting ``include_scalar_shorthand=True``.
    """

    M: AttachMap = normalize_attach_map(attach_map)
    attrs: dict[str, Any] = {attach_attr: M}
    if include_scalar_shorthand and len(M) == 1:
        attrs["attach_index"] = M[0]
    return attrs


def get_attach_map(edge_data: dict[str, Any], *, attach_attr: str = "attach_map") -> AttachMap:
    """Read tuple-valued gluing data, accepting ``attach_index`` as shorthand."""

    if attach_attr in edge_data:
        return normalize_attach_map(edge_data[attach_attr])
    if "attach_index" in edge_data:
        return normalize_attach_map(edge_data["attach_index"])
    raise KeyError(f"edge data is missing {attach_attr!r} and 'attach_index'")


def relabel_to_consecutive_topological(H: nx.DiGraph) -> nx.DiGraph:
    """Return a copy whose node keys are ``0, ..., n-1`` in topological order.

    NetworkX node keys are implementation identifiers, not tree labels. This
    helper keeps encoded graphs visually simple and avoids mixing raw node keys
    with synthetic contraction keys. The returned graph is rebuilt so iteration
    over ``H.nodes`` follows the same topological integer order.
    """

    order = list(nx.topological_sort(H))
    mapping: dict[Hashable, int] = {node: i for i, node in enumerate(order)}
    out = nx.DiGraph()
    out.graph.update(H.graph)
    for old_node in order:
        out.add_node(mapping[old_node], **dict(H.nodes[old_node]))
    for old_u, old_v, data in H.edges(data=True):
        out.add_edge(mapping[old_u], mapping[old_v], **dict(data))
    return out
