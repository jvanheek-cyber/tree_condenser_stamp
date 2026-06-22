"""Exact structural types and fitting-alphabet metadata.

Fitting uses node ``label`` values as opaque symbols.  Decoding uses exact
``type`` values.  A :class:`CompositeType` is an immutable, self-contained
recipe for one transformed occurrence, including component labels/types,
component sizes/root counts, topology, and occurrence-specific attachment
sites.
"""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable, Literal

import networkx as nx

from .exceptions import ValidationError
from .vocabulary import AttachMap, Token, TokenSpec, Vocabulary, is_base_token, normalize_attach_map

CompositeKind = Literal["edge_bpe", "star", "component"]


@dataclass(frozen=True, slots=True)
class CompositeType:
    """Exact recipe for one transformed node occurrence.

    ``label`` is the generic symbol seen by later fitting stages.  The remaining
    fields describe the exact structural variant needed to undo this stage.
    """

    model_id: str
    kind: CompositeKind
    label: Token
    parent: tuple[int, ...]
    component_labels: tuple[Token, ...]
    component_types: tuple[Token, ...]
    component_sizes: tuple[int, ...]
    component_root_counts: tuple[int, ...]
    attach: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.parent)
        if not self.model_id:
            raise ValidationError("CompositeType.model_id must be nonempty.")
        if self.kind not in {"edge_bpe", "star", "component"}:
            raise ValidationError(f"unknown composite kind {self.kind!r}.")
        if not isinstance(self.label, HashableABC):
            raise ValidationError(f"composite label must be hashable; got {self.label!r}.")
        if n == 0:
            raise ValidationError("composite types require at least one component.")
        if not (
            len(self.component_labels)
            == len(self.component_types)
            == len(self.component_sizes)
            == len(self.component_root_counts)
            == n
        ):
            raise ValidationError("all CompositeType component vectors must have equal length.")
        for i, p in enumerate(self.parent):
            if not isinstance(p, int) or isinstance(p, bool) or p < -1 or p >= n or p == i:
                raise ValidationError(f"invalid composite parent[{i}]={p!r}.")
        for i, token in enumerate(self.component_labels):
            if not isinstance(token, HashableABC):
                raise ValidationError(f"component label {i} is not hashable: {token!r}.")
        for i, token in enumerate(self.component_types):
            if not isinstance(token, HashableABC):
                raise ValidationError(f"component type {i} is not hashable: {token!r}.")
        for i, size in enumerate(self.component_sizes):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValidationError(f"component size {i} must be positive; got {size!r}.")
        for i, roots in enumerate(self.component_root_counts):
            if not isinstance(roots, int) or isinstance(roots, bool) or roots <= 0:
                raise ValidationError(
                    f"component root count {i} must be positive; got {roots!r}."
                )
        if not all(isinstance(q, int) and not isinstance(q, bool) for q in self.attach):
            raise ValidationError("CompositeType.attach must be a flat integer tuple.")
        self._check_parent_acyclic()

        expected = sum(
            self.component_root_counts[i]
            for i, p in enumerate(self.parent)
            if p >= 0
        )
        if len(self.attach) != expected:
            raise ValidationError(
                f"composite type stores {len(self.attach)} attachment values; expected {expected}."
            )
        for i, p in enumerate(self.parent):
            if p < 0:
                continue
            parent_sites = self.component_sizes[p]
            bad = [q for q in self.attachment_slice(i) if q < 0 or q >= parent_sites]
            if bad:
                raise ValidationError(
                    f"component {i} has attachment sites {bad!r} outside 0..{parent_sites - 1}."
                )

    @property
    def n_components(self) -> int:
        return len(self.parent)

    @property
    def site_count(self) -> int:
        return sum(self.component_sizes)

    @property
    def root_count(self) -> int:
        return sum(
            self.component_root_counts[i]
            for i, p in enumerate(self.parent)
            if p == -1
        )

    @property
    def root_positions(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.parent) if p == -1)

    def attachment_slice(self, i: int) -> AttachMap:
        if i < 0 or i >= self.n_components:
            raise IndexError(i)
        if self.parent[i] == -1:
            return ()
        start = sum(
            self.component_root_counts[h]
            for h in range(i)
            if self.parent[h] >= 0
        )
        stop = start + self.component_root_counts[i]
        return tuple(self.attach[start:stop])

    def attachment_slices(self) -> tuple[AttachMap, ...]:
        return tuple(self.attachment_slice(i) for i in range(self.n_components))

    def _check_parent_acyclic(self) -> None:
        state = bytearray(len(self.parent))
        for start in range(len(self.parent)):
            if state[start] == 2:
                continue
            path: list[int] = []
            cur = start
            while cur != -1 and state[cur] == 0:
                state[cur] = 1
                path.append(cur)
                cur = self.parent[cur]
            if cur != -1 and state[cur] == 1:
                raise ValidationError("CompositeType parent relation is cyclic.")
            for node in path:
                state[node] = 2


def structural_site_count(type_token: Token, vocab: Vocabulary | None = None) -> int:
    """Return an exact type's number of represented sites."""

    if isinstance(type_token, CompositeType):
        return type_token.site_count
    if is_base_token(type_token):
        return 1
    if vocab is not None and type_token in vocab:
        return vocab.site_count(type_token)
    raise ValidationError(f"cannot determine site count for structural type {type_token!r}.")


def structural_root_count(type_token: Token, vocab: Vocabulary | None = None) -> int:
    """Return an exact type's number of exposed roots."""

    if isinstance(type_token, CompositeType):
        return type_token.root_count
    if is_base_token(type_token):
        return 1
    if vocab is not None and type_token in vocab:
        return vocab.root_count(type_token)
    raise ValidationError(f"cannot determine root count for structural type {type_token!r}.")


def infer_input_alphabet(
    graphs: Sequence[nx.DiGraph],
    *,
    label_attr: str = "label",
    type_attr: str = "type",
    size_attr: str = "size",
    attach_attr: str = "attach_map",
) -> dict[Token, TokenSpec]:
    """Infer and validate fixed size/root metadata for fitting labels.

    A label is a BPE/star symbol and therefore must have one stable expanded
    size and root count throughout a fitted corpus.  Exact attachment variants
    may differ while preserving this specification.
    """

    specs: dict[Token, TokenSpec] = {}
    for graph in graphs:
        roots = [node for node in graph if graph.in_degree(node) == 0]
        if len(roots) != 1:
            raise ValidationError(f"expected one root while inferring alphabet; found {len(roots)}.")
        root = roots[0]
        for node, data in graph.nodes(data=True):
            label = data[label_attr]
            type_token = data[type_attr]
            size = int(data[size_attr])
            try:
                root_count = structural_root_count(type_token)
            except ValidationError:
                if node == root:
                    root_count = 1
                else:
                    parent = next(graph.predecessors(node))
                    root_count = len(normalize_attach_map(graph.edges[parent, node][attach_attr]))
            spec = TokenSpec(site_count=size, root_count=root_count)
            previous = specs.get(label)
            if previous is None:
                specs[label] = spec
            # elif previous != spec:
            #     raise ValidationError(
            #         f"fitting label {label!r} has inconsistent specifications: "
            #         f"{previous!r} versus {spec!r}. Use distinct labels for symbols "
            #         "with different size/root counts."
            #     )
    return specs


def component_super_labels(
    value: Any,
    *,
    component_sizes: Sequence[int],
    flat_uids: Sequence[Any],
) -> tuple[Any, ...]:
    """Return recipe-aligned component provenance with a flat fallback."""

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == len(component_sizes)
    ):
        return tuple(value)

    pieces: list[Any] = []
    cursor = 0
    for size in component_sizes:
        piece = tuple(flat_uids[cursor : cursor + size])
        pieces.append(piece[0] if size == 1 else piece)
        cursor += size
    if cursor != len(flat_uids):
        raise ValidationError(
            f"component sizes consume {cursor} UIDs, but occurrence stores {len(flat_uids)}."
        )
    return tuple(pieces)
