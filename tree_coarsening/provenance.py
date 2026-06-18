"""Flat original-UID provenance utilities for encoded tree nodes."""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Mapping, Sequence
from typing import Any

import networkx as nx

from .exceptions import ValidationError
from .vocabulary import Token, VocabEntry, Vocabulary, is_base_token

PROVENANCE_KEY = "tree_coarsening_provenance"
NODE_ATTRS_KEY = "node_attrs_by_uid"


def is_sequence_like(x: Any) -> bool:
    """Return True for non-string sequences used as provenance containers."""

    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))


def require_hashable_uid(uid: Any) -> Any:
    if not isinstance(uid, HashableABC):
        raise ValidationError(f"UIDs used in super_uids must be hashable; got {uid!r}.")
    return uid


def normalize_super_uids(value: Any) -> tuple[Any, ...]:
    """Normalize a stored ``super_uids`` value to a flat tuple of UIDs."""

    if not is_sequence_like(value):
        raise ValidationError(f"super_uids must be a sequence of UIDs; got {value!r}.")
    out = tuple(value)
    for uid in out:
        require_hashable_uid(uid)
    return out


def split_super_uids(
    token: Token,
    super_uids: Any,
    vocab: Mapping[Token, VocabEntry] | Vocabulary,
) -> tuple[tuple[Any, ...], ...]:
    """Split flat occurrence provenance into recipe-position slices."""

    vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
    uids = normalize_super_uids(super_uids)
    if is_base_token(token):
        if len(uids) != 1:
            raise ValidationError(f"base token {token!r} expects one UID, got {len(uids)}.")
        return (uids,)
    if token not in vocabulary.entries:
        raise ValidationError(f"unknown token {token!r}.")
    entry = vocabulary.entries[token]
    expected = vocabulary.site_count(token)
    if len(uids) != expected:
        raise ValidationError(
            f"token {token!r} expects {expected} super_uids, got {len(uids)}."
        )
    pieces: list[tuple[Any, ...]] = []
    pos = 0
    for child_token in entry.label:
        k = vocabulary.site_count(child_token)
        pieces.append(tuple(uids[pos : pos + k]))
        pos += k
    return tuple(pieces)


def validate_super_uids(
    token: Token,
    super_uids: Any,
    vocab: Mapping[Token, VocabEntry] | Vocabulary,
) -> None:
    vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
    uids = normalize_super_uids(super_uids)
    expected = vocabulary.site_count(token)
    if len(uids) != expected:
        raise ValidationError(
            f"token {token!r} expects {expected} super_uids, got {len(uids)}."
        )


def provenance_from_raw_graph(
    G: nx.DiGraph,
    *,
    uid_attr: str = "uid",
) -> dict[str, Any]:
    """Build a graph-level provenance table from a raw tree."""

    table: dict[Any, dict[str, Any]] = {}
    for node, data in G.nodes(data=True):
        uid = data[uid_attr]
        if uid in table:
            raise ValidationError(f"duplicate UID in provenance table: {uid!r}.")
        table[uid] = dict(data)
    return {NODE_ATTRS_KEY: table, "uid_attr": uid_attr}


def get_node_attrs_by_uid(H: nx.DiGraph) -> dict[Any, dict[str, Any]]:
    """Return graph-level raw-node provenance, or an empty table."""

    prov = H.graph.get(PROVENANCE_KEY, {})
    table = prov.get(NODE_ATTRS_KEY, {}) if isinstance(prov, dict) else {}
    return dict(table)


def copy_graph_provenance(src: nx.DiGraph, dst: nx.DiGraph) -> None:
    """Copy graph-level tree-coarsening provenance metadata."""

    for key, value in src.graph.items():
        dst.graph[key] = value
