"""Validation helpers for raw and encoded tree coarsening graphs."""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Iterable
from numbers import Real
from typing import Any, Hashable

import networkx as nx

from .exceptions import ValidationError
from .provenance import normalize_super_uids, validate_super_uids
from .vocabulary import Vocabulary, normalize_attach_map


def validate_raw_tree(
    G: nx.DiGraph,
    *,
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
    require_uid: bool = False,
) -> Hashable:
    """Validate a raw input graph and return its root node."""

    _validate_basic_tree_shape(G)
    roots = [v for v in G.nodes if G.in_degree(v) == 0]
    root = roots[0]

    seen_uids: set[Any] = set()
    for node, data in G.nodes(data=True):
        if label_attr not in data:
            raise ValidationError(f"node {node!r} is missing {label_attr!r}.")
        if not isinstance(data[label_attr], str):
            raise ValidationError(
                f"node {node!r} has non-string {label_attr!r}: {data[label_attr]!r}."
            )
        if time_attr not in data:
            raise ValidationError(f"node {node!r} is missing {time_attr!r}.")
        if not _is_real_number(data[time_attr]):
            raise ValidationError(
                f"node {node!r} has non-numeric {time_attr!r}: {data[time_attr]!r}."
            )
        if uid_attr in data:
            uid = data[uid_attr]
            if not isinstance(uid, HashableABC):
                raise ValidationError(f"uid for node {node!r} must be hashable; got {uid!r}.")
            if uid in seen_uids:
                raise ValidationError(f"duplicate uid {uid!r}.")
            seen_uids.add(uid)
        elif require_uid:
            raise ValidationError(f"node {node!r} is missing required uid {uid_attr!r}.")
    return root


def validate_encoded_tree(
    H: nx.DiGraph,
    *,
    vocab: Vocabulary | None = None,
    label_attr: str = "label",
    super_uid_attr: str = "super_uids",
    attach_attr: str = "attach_map",
    check_super_uids: bool = True,
) -> Hashable:
    """Validate tree shape plus encoded-token and edge metadata.

    Encoded nodes use ``label`` as the token id. They do not need separate
    ``type`` or encoded-node ``uid`` attributes.
    """

    _validate_basic_tree_shape(H)
    roots = [v for v in H.nodes if H.in_degree(v) == 0]
    root = roots[0]
    vocab = vocab or Vocabulary()

    seen_super_uids: set[Any] = set()
    for node, data in H.nodes(data=True):
        if label_attr not in data:
            raise ValidationError(f"encoded node {node!r} is missing {label_attr!r}.")
        if super_uid_attr not in data:
            raise ValidationError(f"encoded node {node!r} is missing {super_uid_attr!r}.")
        token = data[label_attr]
        if token not in vocab:
            raise ValidationError(f"unknown token {token!r} on encoded node {node!r}.")
        if check_super_uids:
            validate_super_uids(token, data[super_uid_attr], vocab)
            for uid in normalize_super_uids(data[super_uid_attr]):
                if uid in seen_super_uids:
                    raise ValidationError(f"UID {uid!r} appears in more than one encoded node.")
                seen_super_uids.add(uid)

    root_token = H.nodes[root][label_attr]
    root_roots = vocab.root_count(root_token)
    if root_roots != 1:
        raise ValidationError(
            f"encoded root node {root!r} has token {root_token!r} with {root_roots} "
            "exposed roots; a decoded tree root must expose exactly one root."
        )

    for u, v, data in H.edges(data=True):
        if attach_attr not in data:
            if "attach_index" not in data:
                raise ValidationError(f"encoded edge {(u, v)!r} is missing {attach_attr!r}.")
            data[attach_attr] = normalize_attach_map(data["attach_index"])
        else:
            data[attach_attr] = normalize_attach_map(data[attach_attr])

        M = data[attach_attr]
        parent_token = H.nodes[u][label_attr]
        child_token = H.nodes[v][label_attr]
        expected_len = vocab.root_count(child_token)
        parent_sites = vocab.site_count(parent_token)
        if len(M) != expected_len:
            raise ValidationError(
                f"edge {(u, v)!r} has attach_map length {len(M)}, expected {expected_len}."
            )
        if any(k < 0 or k >= parent_sites for k in M):
            raise ValidationError(
                f"edge {(u, v)!r} has attach_map {M}, but parent has {parent_sites} sites."
            )
    return root


def copy_with_uids(G: nx.DiGraph, *, uid_attr: str = "uid") -> nx.DiGraph:
    """Return a copy in which every raw node has a unique UID."""

    H = G.copy(as_view=False)
    seen: set[Any] = set()
    for node, data in H.nodes(data=True):
        uid = data.get(uid_attr, node)
        if not isinstance(uid, HashableABC):
            raise ValidationError(f"uid for node {node!r} must be hashable; got {uid!r}.")
        if uid in seen:
            raise ValidationError(f"duplicate uid after fallback assignment: {uid!r}")
        data[uid_attr] = uid
        seen.add(uid)
    return H


def deterministic_node_order(
    G: nx.DiGraph,
    nodes: Iterable[Hashable],
    *,
    label_attr: str = "label",
    time_attr: str = "time",
    uid_attr: str = "uid",
) -> list[Hashable]:
    """Sort nodes deterministically using time, label, UID, then node key."""

    def key(node: Hashable) -> tuple[str, str, str, str]:
        data = G.nodes[node]
        return (
            repr(data.get(time_attr, "")),
            repr(data.get(label_attr, "")),
            repr(data.get(uid_attr, "")),
            repr(node),
        )

    return sorted(nodes, key=key)


def _validate_basic_tree_shape(G: nx.DiGraph) -> None:
    if not isinstance(G, nx.DiGraph) or G.is_multigraph():
        raise ValidationError("graph must be a networkx.DiGraph, not a multigraph.")
    n = G.number_of_nodes()
    if n == 0:
        raise ValidationError("tree must contain at least one node.")
    if G.number_of_edges() != n - 1:
        raise ValidationError(
            f"tree with {n} nodes must have {n - 1} edges; got {G.number_of_edges()}."
        )
    if not nx.is_connected(G.to_undirected(as_view=True)):
        raise ValidationError("underlying undirected graph must be connected.")
    roots = [v for v in G.nodes if G.in_degree(v) == 0]
    if len(roots) != 1:
        raise ValidationError(f"expected exactly one root; found {len(roots)}.")
    root = roots[0]
    bad = [v for v in G.nodes if v != root and G.in_degree(v) != 1]
    if bad:
        raise ValidationError(f"every non-root node must have in-degree 1; bad nodes: {bad!r}.")
    if not nx.is_directed_acyclic_graph(G):
        raise ValidationError("directed tree must be acyclic.")


def _is_real_number(x: Any) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)
