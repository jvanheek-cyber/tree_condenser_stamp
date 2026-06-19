"""Canonical graph schema shared by all tree coarseners.

The public object remains ``networkx.DiGraph``.  This module centralizes the
small normalization layer that lets raw trees and previously transformed trees
be consumed through the same fitting interface.

Every normalized node has:

``label``
    Hashable symbol used by statistical fitting.
``type``
    Exact structural/decoder type. Several exact structural variants may share
    one fitting ``label``.
``size``
    Positive number of represented original vertices/sites.
``time``
    Numeric representative timestamp.  Contractions use the maximum component
    timestamp.
``super_label``
    Recipe-aligned provenance payload. Raw nodes use their UID; transformed
    occurrences preserve the nested component structure needed by stage-local
    decoding.
``super_uids``
    Backward-compatible flat UID tuple used by the current decoder.

Edges may omit ``attach_map`` only for raw input, in which case ``(0,)`` is
materialized on the normalized copy.
"""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Sequence
from numbers import Integral, Real
from typing import Any, TypeAlias

import networkx as nx

from .exceptions import ValidationError
from .provenance import get_node_attrs_by_uid, normalize_super_uids
from .vocabulary import Token, base_token, normalize_attach_map

GraphInput: TypeAlias = nx.DiGraph | Sequence[nx.DiGraph]
RAW_INPUT_FLAG = "tree_coarsening_input_was_raw"


def as_graph_list(graphs: GraphInput, *, argument_name: str = "graphs") -> tuple[list[nx.DiGraph], bool]:
    """Normalize one graph or a graph sequence to ``(list, was_single)``."""

    if isinstance(graphs, nx.DiGraph):
        return [graphs], True
    if isinstance(graphs, (str, bytes, bytearray)):
        raise TypeError(f"{argument_name} must be a DiGraph or a sequence of DiGraphs.")
    try:
        out = list(graphs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"{argument_name} must be a DiGraph or a sequence of DiGraphs."
        ) from exc
    if not out:
        raise ValueError(f"{argument_name} requires at least one graph.")
    if not all(isinstance(graph, nx.DiGraph) and not graph.is_multigraph() for graph in out):
        raise TypeError(f"every element of {argument_name} must be a networkx.DiGraph.")
    return out, False


def flatten_super_label(value: Any) -> tuple[Any, ...]:
    """Flatten a nested provenance payload into a left-to-right UID tuple."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[Any] = []
        for item in value:
            out.extend(flatten_super_label(item))
        return tuple(out)
    if not isinstance(value, HashableABC):
        raise ValidationError(f"super_label leaves must be hashable; got {value!r}.")
    return (value,)


def node_structural_type(
    data: dict[str, Any],
    *,
    label_attr: str = "label",
    type_attr: str = "type",
) -> Token:
    """Return a node's exact structural type with raw-input fallback."""

    if type_attr in data:
        token = data[type_attr]
    else:
        label = data[label_attr]
        # Raw string labels are represented structurally as implicit base tokens;
        # transformed hashable labels already denote their own stage symbol/type.
        token = base_token(label) if isinstance(label, str) else label
    if not isinstance(token, HashableABC):
        raise ValidationError(f"node type must be hashable; got {token!r}.")
    return token


def _infer_super_fields(
    node: Any,
    data: dict[str, Any],
    *,
    uid_attr: str,
    super_label_attr: str,
    super_uid_attr: str,
) -> tuple[Any, tuple[Any, ...]]:
    # Prefer the unambiguous flat compatibility field when both are present.
    # A UID may itself be a tuple, which makes a scalar tuple-valued
    # ``super_label`` impossible to distinguish from nested provenance without
    # its accompanying ``super_uids`` field.
    if super_uid_attr in data:
        flat = normalize_super_uids(data[super_uid_attr])
        super_label = data.get(
            super_label_attr,
            flat[0] if len(flat) == 1 else flat,
        )
    elif super_label_attr in data:
        super_label = data[super_label_attr]
        flat = flatten_super_label(super_label)
    else:
        uid = data.get(uid_attr, node)
        if not isinstance(uid, HashableABC):
            raise ValidationError(f"uid for node {node!r} must be hashable; got {uid!r}.")
        super_label = uid
        flat = (uid,)
    return super_label, flat


def _infer_time(
    G: nx.DiGraph,
    data: dict[str, Any],
    super_uids: tuple[Any, ...],
    *,
    time_attr: str,
) -> float:
    if time_attr in data:
        value = data[time_attr]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValidationError(f"node time must be numeric; got {value!r}.")
        return float(value)

    provenance = get_node_attrs_by_uid(G)
    values: list[float] = []
    for uid in super_uids:
        attrs = provenance.get(uid)
        if attrs is None or time_attr not in attrs:
            raise ValidationError(
                f"node is missing {time_attr!r} and provenance has no time for UID {uid!r}."
            )
        value = attrs[time_attr]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValidationError(f"provenance time for UID {uid!r} is not numeric: {value!r}.")
        values.append(float(value))
    if not values:
        raise ValidationError("cannot infer a representative time from empty provenance.")
    return max(values)


def normalize_coarsenable_tree(
    G: nx.DiGraph,
    *,
    label_attr: str = "label",
    type_attr: str = "type",
    size_attr: str = "size",
    time_attr: str = "time",
    uid_attr: str = "uid",
    super_label_attr: str = "super_label",
    super_uid_attr: str = "super_uids",
    attach_attr: str = "attach_map",
    copy: bool = True,
) -> nx.DiGraph:
    """Return a canonical fitting-tree view of raw or transformed input.

    The function deliberately performs no vocabulary lookup and never requires
    attachment metadata for fitting.  Missing raw-edge attachment maps are
    materialized only so the same returned graph can later be transformed.
    """

    was_raw = G.graph.get(
        RAW_INPUT_FLAG,
        all(
            type_attr not in data
            and size_attr not in data
            and super_label_attr not in data
            and super_uid_attr not in data
            for _node, data in G.nodes(data=True)
        ),
    )
    H = G.copy(as_view=False) if copy else G
    H.graph[RAW_INPUT_FLAG] = bool(was_raw)
    for node, data in H.nodes(data=True):
        if label_attr not in data:
            raise ValidationError(f"node {node!r} is missing {label_attr!r}.")
        label = data[label_attr]
        if not isinstance(label, HashableABC):
            raise ValidationError(f"node {node!r} has non-hashable label {label!r}.")

        super_label, super_uids = _infer_super_fields(
            node,
            data,
            uid_attr=uid_attr,
            super_label_attr=super_label_attr,
            super_uid_attr=super_uid_attr,
        )
        size = data.get(size_attr, len(super_uids))
        if not isinstance(size, Integral) or isinstance(size, bool) or int(size) <= 0:
            raise ValidationError(f"node {node!r} has invalid positive integer size {size!r}.")
        size = int(size)
        if size != len(super_uids):
            raise ValidationError(
                f"node {node!r} has size={size}, but its provenance contains "
                f"{len(super_uids)} original UIDs."
            )

        data[type_attr] = node_structural_type(data, label_attr=label_attr, type_attr=type_attr)
        data[size_attr] = size
        data[time_attr] = _infer_time(H, data, super_uids, time_attr=time_attr)
        data[super_label_attr] = super_label
        data[super_uid_attr] = tuple(super_uids)
        data.setdefault(uid_attr, node)

    for _u, _v, data in H.edges(data=True):
        if attach_attr in data:
            data[attach_attr] = normalize_attach_map(data[attach_attr])
        elif "attach_index" in data:
            data[attach_attr] = normalize_attach_map(data["attach_index"])
        else:
            data[attach_attr] = (0,)
    return H


def max_component_time(*values: Real) -> float:
    """Canonical timestamp aggregation for contractions."""

    if not values:
        raise ValueError("max_component_time requires at least one value.")
    if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
        raise ValidationError(f"all component times must be numeric; got {values!r}.")
    return float(max(values))


def encoded_node_attrs(
    *,
    label: Any,
    type_token: Any,
    size: int,
    time: Real,
    super_label: Any,
    super_uids: Sequence[Any],
    label_attr: str = "label",
    type_attr: str = "type",
    size_attr: str = "size",
    time_attr: str = "time",
    super_label_attr: str = "super_label",
    super_uid_attr: str = "super_uids",
) -> dict[str, Any]:
    """Construct the canonical node-attribute payload emitted by coarseners."""

    uids = tuple(super_uids)
    if not isinstance(label, HashableABC):
        raise ValidationError(f"encoded label must be hashable; got {label!r}.")
    if not isinstance(type_token, HashableABC):
        raise ValidationError(f"encoded type must be hashable; got {type_token!r}.")
    if not isinstance(size, Integral) or isinstance(size, bool) or int(size) <= 0:
        raise ValidationError(f"encoded size must be a positive integer; got {size!r}.")
    if int(size) != len(uids):
        raise ValidationError(
            f"encoded size={size} does not match {len(uids)} represented UIDs."
        )
    if not isinstance(time, Real) or isinstance(time, bool):
        raise ValidationError(f"encoded time must be numeric; got {time!r}.")
    return {
        label_attr: label,
        type_attr: type_token,
        size_attr: int(size),
        time_attr: float(time),
        super_label_attr: super_label,
        super_uid_attr: uids,
    }
