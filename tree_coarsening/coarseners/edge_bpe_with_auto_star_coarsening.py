"""Edge BPE with automatic star coarsening for directed labeled trees.

This variant selects which label pair to merge exactly like
:mod:`tree_coarsening.coarseners.edge_bpe`: at each step the highest-scoring
``(parent_label, child_label)`` pair is chosen from raw matching-edge counts,
using the same configurable ``pair_score``.

It differs in the *transform* step.  Where plain edge BPE contracts a single
parent/child pair at a time, this coarsener inspects each parent and contracts
*all* of its children that carry the selected child label into the parent at
once -- a labeled "star" merge restricted to the candidate children of the
selected pair.  Every group contracted by one rule shares a single fitting
label regardless of how many children it absorbs; the contracted arity is
intentionally not recorded.

As in plain edge BPE, fitting consumes only topology plus node ``label``,
``size``, and ``time``; attachment maps play no role in rule learning.  During
transformation the actual occurrence attachment maps are retained in an exact
:class:`CompositeType` so stage-local decoding remains lossless.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..nx_io import edge_attach_attrs
from ..provenance import (
    PROVENANCE_KEY,
    get_node_attrs_by_uid,
    provenance_from_raw_graph,
)
from ..schema import (
    RAW_INPUT_FLAG,
    encoded_node_attrs,
    max_component_time,
    normalize_coarsenable_tree,
)
from ..stage_decoder import StructuralStageDecoder
from ..structural import CompositeType, infer_input_alphabet, structural_root_count
from ..validation import validate_coarsenable_tree, validate_encoded_tree
from ..vocabulary import Token, Vocabulary, normalize_attach_map

EdgeKey = tuple[int, int]
PairScoreFunction = Callable[[int, int, int, int, int], float]
PairScoreName = Literal["count", "normalized", "size_weighted"]
PairScore = PairScoreName | PairScoreFunction


def count_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return the ordinary unweighted BPE score ``N(A, B)``."""

    del n_a, n_b, s_a, s_b
    return float(n_ab)


def normalized_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return ``N(A,B) / sqrt(N(A) N(B))``."""

    del s_a, s_b
    if n_a <= 0 or n_b <= 0:
        raise ValidationError(
            "normalized pair score requires positive endpoint occurrence counts."
        )
    return float(n_ab) / math.sqrt(float(n_a) * float(n_b))


def size_weighted_pair_score(
    n_ab: int,
    n_a: int,
    n_b: int,
    s_a: int,
    s_b: int,
) -> float:
    """Return ``N(A,B) * (S(A) + S(B))``."""

    del n_a, n_b
    return float(n_ab) * float(s_a + s_b)


_BUILTIN_PAIR_SCORES: dict[PairScoreName, PairScoreFunction] = {
    "count": count_pair_score,
    "normalized": normalized_pair_score,
    "size_weighted": size_weighted_pair_score,
}


@dataclass(frozen=True, slots=True)
class _PairSelection:
    key: EdgeKey
    count: int
    parent_count: int
    child_count: int
    parent_size: int
    child_size: int
    score: float


def edge_star_token(rank: int) -> tuple[str, int]:
    """Stable fitting label for the ``rank``-th star merge.

    Every group contracted at a given ``rank`` shares this label regardless of
    how many children it absorbs: star-condensed nodes are identified by label
    alone, ignoring arity.
    """

    if rank < 0:
        raise ValidationError("edge-star rank must be nonnegative.")
    return ("edge_star", int(rank))


@dataclass(frozen=True, slots=True)
class EdgeStarRule:
    """One fitted star-merge rule for a selected ``(parent_label, child_label)`` pair.

    ``count`` is the raw number of matching live edges at selection time, used
    only for pair selection.  ``token`` is the single fitting label assigned to
    every contracted group of this rule; the number of children a group absorbs
    (its arity) is intentionally not recorded.
    """

    rank: int
    token: Token
    parent_label: Token
    child_label: Token
    count: int
    score: float | None = None
    parent_count: int | None = None
    child_count: int | None = None
    parent_size: int | None = None
    child_size: int | None = None

    # Compatibility aliases used by earlier examples/tests.
    @property
    def parent_token(self) -> Token:
        return self.parent_label

    @property
    def child_token(self) -> Token:
        return self.child_label


@dataclass(slots=True)
class _TokenCodec:
    token_to_id: dict[Token, int] = field(default_factory=dict)
    id_to_token: list[Token] = field(default_factory=list)
    sort_key_by_id: list[str] = field(default_factory=list)

    def intern(self, token: Token) -> int:
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)
        self.sort_key_by_id.append(repr(token))
        return token_id

    def decode(self, token_id: int) -> Token:
        return self.id_to_token[token_id]

    def sort_key(self, token_id: int) -> str:
        return self.sort_key_by_id[token_id]


@dataclass(slots=True)
class _UidRope:
    """Append-only n-ary UID provenance rope used only while transforming.

    A star merge concatenates more than two components, so each rope node stores
    an ordered list of child references.  Flattening occurs once, when the final
    encoded NetworkX graph is produced.
    """

    leaves: list[tuple[Any, ...]]
    children: list[tuple[int, ...]] = field(default_factory=list)

    def merge(self, parts: Sequence[int]) -> int:
        ref = len(self.leaves) + len(self.children)
        self.children.append(tuple(parts))
        return ref

    def flatten(self, ref: int) -> tuple[Any, ...]:
        leaf_count = len(self.leaves)
        out: list[Any] = []
        stack = [ref]
        while stack:
            current = stack.pop()
            if current < leaf_count:
                out.extend(self.leaves[current])
                continue
            merge_index = current - leaf_count
            if merge_index < 0 or merge_index >= len(self.children):
                raise RuntimeError(f"invalid provenance-rope reference {current!r}.")
            for part in reversed(self.children[merge_index]):
                stack.append(part)
        return tuple(out)


@dataclass(slots=True)
class _OutputContext:
    label_attr: str
    type_attr: str
    size_attr: str
    time_attr: str
    super_label_attr: str
    super_uid_attr: str
    attach_attr: str
    model_id: str
    provenance: dict[str, Any]
    uid_rope: _UidRope


def _bump_count(counts: Counter[EdgeKey], key: EdgeKey, delta: int) -> None:
    value = counts.get(key, 0) + delta
    if value < 0:
        raise RuntimeError(f"edge-pair count for {key!r} became negative.")
    if value == 0:
        counts.pop(key, None)
    else:
        counts[key] = value


def _initial_label_statistics(
    states: Sequence["_CompactEdgeTree"],
    codec: _TokenCodec,
    vocab: Vocabulary,
) -> tuple[list[int], list[int]]:
    """Return dense occurrence-count and fixed-size arrays by label id."""

    label_counts = [0] * len(codec.id_to_token)
    for state in states:
        for node, is_alive in enumerate(state.alive):
            if is_alive:
                label_counts[state.label[node]] += 1
    label_sizes = [
        vocab.site_count(codec.decode(i)) for i in range(len(codec.id_to_token))
    ]
    return label_counts, label_sizes


def _set_new_label_statistics(
    label_counts: list[int],
    label_sizes: list[int],
    *,
    label_id: int,
    size: int,
) -> None:
    """Ensure dense arrays contain ``label_id`` and register its fixed size."""

    while len(label_counts) <= label_id:
        label_counts.append(0)
        label_sizes.append(0)
    label_counts[label_id] = 0
    label_sizes[label_id] = size


@dataclass(slots=True)
class _CompactEdgeTree:
    """Mutable array-backed state shared by fitting and transformation.

    Fit-time state contains only ``parent``, ``children``, integer ``label``,
    ``size``, ``time``, liveness, and the label-pair index.  Output-only exact
    types, nested provenance, UIDs, root counts, and attachment maps are
    retained only when ``capture_output=True``.
    """

    parent: list[int]
    children: list[list[int]]
    label: list[int]
    size: list[int]
    time: list[float]
    alive: list[bool]
    codec: _TokenCodec
    vocab: Vocabulary
    edge_index: dict[EdgeKey, set[int]] = field(default_factory=lambda: defaultdict(set))

    output: _OutputContext | None = None
    type_token: list[Token] | None = None
    super_label: list[Any] | None = None
    uid_ref: list[int] | None = None
    root_count: list[int] | None = None
    attach_to_parent: list[tuple[int, ...]] | None = None

    @classmethod
    def from_graph(
        cls,
        G: nx.DiGraph,
        *,
        codec: _TokenCodec,
        vocab: Vocabulary,
        label_attr: str = "label",
        type_attr: str = "type",
        size_attr: str = "size",
        time_attr: str = "time",
        uid_attr: str = "uid",
        super_label_attr: str = "super_label",
        super_uid_attr: str = "super_uids",
        attach_attr: str = "attach_map",
        model_id: str = "",
        pair_counts: Counter[EdgeKey] | None = None,
        capture_output: bool = False,
        build_edge_index: bool = True,
    ) -> "_CompactEdgeTree":
        roots = [node for node in G if G.in_degree(node) == 0]
        if len(roots) != 1:
            raise ValidationError(f"expected exactly one root; found {len(roots)}.")

        root = roots[0]
        order: list[Any] = [root]
        seen = {root}
        parent = [-1]
        children: list[list[int]] = [[]]

        def child_key(child: Any) -> tuple[Any, ...]:
            data = G.nodes[child]
            return (
                float(data[time_attr]),
                repr(data[label_attr]),
                repr(data.get(uid_attr, child)),
                repr(child),
            )

        cursor = 0
        while cursor < len(order):
            node = order[cursor]
            for child in sorted(G.successors(node), key=child_key):
                if child in seen:
                    raise ValidationError(f"node {child!r} is reachable more than once.")
                child_i = len(order)
                seen.add(child)
                order.append(child)
                parent.append(cursor)
                children.append([])
                children[cursor].append(child_i)
            cursor += 1
        if len(order) != G.number_of_nodes():
            raise ValidationError("not every node is reachable from the directed root.")

        labels = [codec.intern(G.nodes[node][label_attr]) for node in order]
        sizes = [int(G.nodes[node][size_attr]) for node in order]
        times = [float(G.nodes[node][time_attr]) for node in order]
        alive = [True] * len(order)

        output: _OutputContext | None = None
        types: list[Token] | None = None
        super_labels: list[Any] | None = None
        uid_ref: list[int] | None = None
        roots_per_node: list[int] | None = None
        attachments: list[tuple[int, ...]] | None = None

        if capture_output:
            types = [G.nodes[node][type_attr] for node in order]
            super_labels = [G.nodes[node][super_label_attr] for node in order]
            uid_leaves = [tuple(G.nodes[node][super_uid_attr]) for node in order]
            uid_rope = _UidRope(uid_leaves)
            uid_ref = list(range(len(order)))
            roots_per_node = []
            attachments = []
            for i, node in enumerate(order):
                p = parent[i]
                if p == -1:
                    roots_per_node.append(structural_root_count(types[i], vocab))
                    attachments.append(())
                else:
                    edge_map = normalize_attach_map(
                        G.edges[order[p], node][attach_attr]
                    )
                    roots_per_node.append(len(edge_map))
                    attachments.append(edge_map)

            provenance = G.graph.get(PROVENANCE_KEY)
            if isinstance(provenance, dict) and get_node_attrs_by_uid(G):
                provenance_payload = provenance
            else:
                provenance_payload = provenance_from_raw_graph(G, uid_attr=uid_attr)
            output = _OutputContext(
                label_attr=label_attr,
                type_attr=type_attr,
                size_attr=size_attr,
                time_attr=time_attr,
                super_label_attr=super_label_attr,
                super_uid_attr=super_uid_attr,
                attach_attr=attach_attr,
                model_id=model_id,
                provenance=provenance_payload,
                uid_rope=uid_rope,
            )

        state = cls(
            parent=parent,
            children=children,
            label=labels,
            size=sizes,
            time=times,
            alive=alive,
            codec=codec,
            vocab=vocab,
            output=output,
            type_token=types,
            super_label=super_labels,
            uid_ref=uid_ref,
            root_count=roots_per_node,
            attach_to_parent=attachments,
        )
        if build_edge_index:
            state.rebuild_edge_index(pair_counts=pair_counts)
        return state

    def rebuild_edge_index(self, *, pair_counts: Counter[EdgeKey] | None = None) -> None:
        self.edge_index = defaultdict(set)
        for child in range(len(self.parent)):
            if self._edge_is_live(child):
                self._add_edge(child, pair_counts=pair_counts)

    def _edge_is_live(self, child: int) -> bool:
        if child < 0 or child >= len(self.alive) or not self.alive[child]:
            return False
        p = self.parent[child]
        return p >= 0 and p < len(self.alive) and self.alive[p]

    def _edge_key_unchecked(self, child: int) -> EdgeKey:
        p = self.parent[child]
        return (self.label[p], self.label[child])

    def _edge_key(self, child: int) -> EdgeKey:
        if not self._edge_is_live(child):
            raise ValidationError(f"node {child!r} does not have a live incoming edge.")
        return self._edge_key_unchecked(child)

    def _add_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        self.edge_index[key].add(child)
        if pair_counts is not None:
            _bump_count(pair_counts, key, 1)

    def _remove_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        bucket = self.edge_index.get(key)
        if bucket is None or child not in bucket:
            raise RuntimeError(f"live edge for child {child!r} is missing from its bucket.")
        bucket.remove(child)
        if not bucket:
            self.edge_index.pop(key, None)
        if pair_counts is not None:
            _bump_count(pair_counts, key, -1)

    def _edge_sort_key(self, child: int) -> tuple[float, float, int, int]:
        p = self.parent[child]
        return (self.time[child], self.time[p], p, child)

    def contract_star_pair(
        self,
        key: EdgeKey,
        *,
        new_label: int,
        rule_token: Token,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> tuple[int, int]:
        """Contract all matching children into their parents as star groups.

        Every parent with one or more live children matching ``key`` has all of
        those children contracted into it at once and relabeled to ``rule_token``,
        regardless of how many children the group contains.  Returns
        ``(group_count, child_count)``: the number of contracted groups and the
        total number of children absorbed across them.
        """

        bucket = self.edge_index.get(key)
        if not bucket:
            return (0, 0)

        # Snapshot candidate children grouped by their (live) parent.  Children
        # are visited in a deterministic order so the contracted recipe order is
        # reproducible.
        groups: dict[int, list[int]] = defaultdict(list)
        for child in sorted(bucket, key=self._edge_sort_key):
            if not self._edge_is_live(child):
                continue
            if self._edge_key_unchecked(child) != key:
                continue
            groups[self.parent[child]].append(child)

        group_count = 0
        child_count = 0
        # Contract parents in a deterministic order.  When the selected pair is a
        # self-pair (parent_label == child_label) a node can appear both as a
        # parent and as a member of its own parent's group; re-checking liveness
        # keeps overlapping chains correct without double-contracting a node.
        for parent_node in sorted(groups, key=lambda p: (self.time[p], p)):
            if not self.alive[parent_node]:
                continue
            members = [
                child
                for child in groups[parent_node]
                if self._edge_is_live(child)
                and self.parent[child] == parent_node
                and self._edge_key_unchecked(child) == key
            ]
            if not members:
                continue
            self._contract_star_group(
                parent_node,
                members,
                new_label=new_label,
                rule_token=rule_token,
                pair_counts=pair_counts,
            )
            group_count += 1
            child_count += len(members)
        return (group_count, child_count)

    def _contract_star_group(
        self,
        parent_node: int,
        child_nodes: Sequence[int],
        *,
        new_label: int,
        rule_token: Token,
        pair_counts: Counter[EdgeKey] | None,
    ) -> None:
        if not child_nodes:
            raise ValidationError("a star group must contain at least one child.")
        for child_node in child_nodes:
            if not self._edge_is_live(child_node) or self.parent[child_node] != parent_node:
                raise ValidationError("attempted to contract a non-live edge occurrence.")

        grandparent = self.parent[parent_node]
        parent_size = self.size[parent_node]
        child_set = set(child_nodes)

        # Remove every edge whose key disappears or changes: the parent's
        # incoming edge, every old parent edge (contracted children plus other
        # children), and the outgoing edges of each contracted child.
        if grandparent != -1:
            self._remove_edge(parent_node, pair_counts=pair_counts)
        retained: list[int] = []
        for current in self.children[parent_node]:
            self._remove_edge(current, pair_counts=pair_counts)
            if current not in child_set:
                retained.append(current)
        for child_node in child_nodes:
            for grandchild in self.children[child_node]:
                self._remove_edge(grandchild, pair_counts=pair_counts)

        # Build the exact occurrence type before mutating parent/child arrays.
        if self.output is not None:
            if (
                self.type_token is None
                or self.super_label is None
                or self.uid_ref is None
                or self.root_count is None
                or self.attach_to_parent is None
            ):
                raise RuntimeError("output-enabled compact state is incomplete.")
            component_labels = [self.codec.decode(self.label[parent_node])]
            component_types = [self.type_token[parent_node]]
            component_sizes = [parent_size]
            component_roots = [self.root_count[parent_node]]
            parents_vec = [-1]
            attach_flat: list[int] = []
            for child_node in child_nodes:
                component_labels.append(self.codec.decode(self.label[child_node]))
                component_types.append(self.type_token[child_node])
                component_sizes.append(self.size[child_node])
                component_roots.append(self.root_count[child_node])
                parents_vec.append(0)
                attach_flat.extend(self.attach_to_parent[child_node])
            exact_type = CompositeType(
                model_id=self.output.model_id,
                kind="edge_bpe",
                label=rule_token,
                parent=tuple(parents_vec),
                component_labels=tuple(component_labels),
                component_types=tuple(component_types),
                component_sizes=tuple(component_sizes),
                component_root_counts=tuple(component_roots),
                attach=tuple(attach_flat),
            )
            self.type_token[parent_node] = exact_type
            self.super_label[parent_node] = tuple(
                [self.super_label[parent_node]]
                + [self.super_label[child_node] for child_node in child_nodes]
            )
            self.uid_ref[parent_node] = self.output.uid_rope.merge(
                [self.uid_ref[parent_node]]
                + [self.uid_ref[child_node] for child_node in child_nodes]
            )
            for child_node in child_nodes:
                self.uid_ref[child_node] = -1
            # The new token inherits the parent's exposed roots and incoming map.

        # Relabel the surviving parent position and accumulate size/time.
        merged_time = self.time[parent_node]
        total_child_size = 0
        for child_node in child_nodes:
            merged_time = max_component_time(merged_time, self.time[child_node])
            total_child_size += self.size[child_node]
        self.label[parent_node] = new_label
        self.size[parent_node] = parent_size + total_child_size
        self.time[parent_node] = merged_time

        # Reparent grandchildren, offsetting their attachment site into the new
        # token's coordinate space.  The offset advances by each child's site
        # count so the layout matches the recipe (P, C_1, ..., C_k).
        new_children = retained
        site_offset = parent_size
        for child_node in child_nodes:
            child_size = self.size[child_node]
            for grandchild in self.children[child_node]:
                self.parent[grandchild] = parent_node
                if self.attach_to_parent is not None:
                    self.attach_to_parent[grandchild] = tuple(
                        site_offset + q for q in self.attach_to_parent[grandchild]
                    )
                new_children.append(grandchild)
            site_offset += child_size

        for child_node in child_nodes:
            self.alive[child_node] = False
            self.parent[child_node] = -1
            self.children[child_node] = []
            if self.attach_to_parent is not None:
                self.attach_to_parent[child_node] = ()

        self.children[parent_node] = new_children

        if grandparent != -1:
            self._add_edge(parent_node, pair_counts=pair_counts)
        for current in new_children:
            self._add_edge(current, pair_counts=pair_counts)

    def to_networkx(self, *, validate: bool = True) -> nx.DiGraph:
        if (
            self.output is None
            or self.type_token is None
            or self.super_label is None
            or self.uid_ref is None
            or self.attach_to_parent is None
        ):
            raise RuntimeError("fit-time compact states cannot be emitted as NetworkX.")
        context = self.output
        live = [node for node, keep in enumerate(self.alive) if keep]
        mapping = {old: new for new, old in enumerate(live)}
        H = nx.DiGraph()
        H.graph[PROVENANCE_KEY] = context.provenance
        H.graph[RAW_INPUT_FLAG] = False
        H.graph["tree_coarsening_schema"] = {
            "schema_version": "0.3",
            "model_id": context.model_id,
            "node_label_semantics": "fit symbol",
            "node_type_semantics": "exact structural variant",
        }
        for old in live:
            uids = context.uid_rope.flatten(self.uid_ref[old])
            H.add_node(
                mapping[old],
                **encoded_node_attrs(
                    label=self.codec.decode(self.label[old]),
                    type_token=self.type_token[old],
                    size=self.size[old],
                    time=self.time[old],
                    super_label=self.super_label[old],
                    super_uids=uids,
                    label_attr=context.label_attr,
                    type_attr=context.type_attr,
                    size_attr=context.size_attr,
                    time_attr=context.time_attr,
                    super_label_attr=context.super_label_attr,
                    super_uid_attr=context.super_uid_attr,
                ),
            )
        for old_child in live:
            old_parent = self.parent[old_child]
            if old_parent == -1:
                continue
            H.add_edge(
                mapping[old_parent],
                mapping[old_child],
                **edge_attach_attrs(
                    self.attach_to_parent[old_child],
                    attach_attr=context.attach_attr,
                ),
            )
        if validate:
            validate_encoded_tree(
                H,
                vocab=self.vocab,
                label_attr=context.label_attr,
                type_attr=context.type_attr,
                size_attr=context.size_attr,
                time_attr=context.time_attr,
                super_label_attr=context.super_label_attr,
                super_uid_attr=context.super_uid_attr,
                attach_attr=context.attach_attr,
            )
        return H


@dataclass
class EdgeBPEWithAutoStarEncoder(TreeEncoder):
    """Apply fitted star-merge rules with occurrence-specific exact types."""

    edge_rules: tuple[EdgeStarRule, ...] = ()

    def encode(
        self, G: nx.DiGraph, *, validate: bool = True, max_steps: int | None = None
    ) -> nx.DiGraph:
        G = normalize_coarsenable_tree(
            G,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_label_attr=self.super_label_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            copy=True,
        )
        if validate:
            validate_coarsenable_tree(
                G,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
            )

        codec = _TokenCodec()
        for label in sorted(self.vocab.symbols, key=repr):
            codec.intern(label)
        for rule in self.edge_rules:
            codec.intern(rule.parent_label)
            codec.intern(rule.child_label)
            codec.intern(rule.token)

        state = _CompactEdgeTree.from_graph(
            G,
            codec=codec,
            vocab=self.vocab,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_label_attr=self.super_label_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            model_id=self.model_id,
            capture_output=True,
        )
        for i, rule in enumerate(self.edge_rules):
            if max_steps is not None and i >= max_steps:
                break
            key = (codec.intern(rule.parent_label), codec.intern(rule.child_label))
            state.contract_star_pair(
                key,
                new_label=codec.intern(rule.token),
                rule_token=rule.token,
                pair_counts=None,
            )
        return state.to_networkx(validate=validate)


class EdgeBPEWithAutoStarCoarsener(TreeCoarsener):
    """Edge BPE that contracts every matching child of a selected pair at once.

    Pair selection is identical to :class:`EdgeBPECoarsener`: at each step the
    score of a pair is computed from its raw number of matching
    ``(parent_label, child_label)`` edges using ``pair_score``.  The transform
    differs.  After a rule is selected, every parent that has one or more
    matching children has *all* of them contracted together with the parent into
    a single new node.  Every group contracted by one rule shares a single
    fitting label regardless of how many children it absorbs; the contracted
    arity is intentionally not recorded.
    """

    def __init__(
        self,
        *,
        num_merges: int | None = None,
        min_pair_count: int = 2,
        pair_score: PairScore = "count",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must be None or nonnegative.")
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be at least 1.")
        if isinstance(pair_score, str):
            if pair_score not in _BUILTIN_PAIR_SCORES:
                allowed = ", ".join(sorted(_BUILTIN_PAIR_SCORES))
                raise ValueError(f"pair_score must be one of {allowed}, or a callable.")
            pair_score_name: PairScoreName | None = pair_score
            pair_score_function = _BUILTIN_PAIR_SCORES[pair_score]
        elif callable(pair_score):
            pair_score_name = None
            pair_score_function = pair_score
        else:
            raise TypeError("pair_score must be a built-in score name or a callable.")
        self.num_merges = num_merges
        self.min_pair_count = min_pair_count
        self.pair_score = pair_score
        self.pair_score_name_: PairScoreName | None = pair_score_name
        self._pair_score_function: PairScoreFunction = pair_score_function
        self.pair_score_display_name_ = (
            pair_score_name
            if pair_score_name is not None
            else getattr(pair_score_function, "__name__", "custom")
        )
        self.history_: list[dict[str, Any]] = []

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        input_alphabet = infer_input_alphabet(
            graphs,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            attach_attr=self.attach_attr,
        )
        vocab = Vocabulary(symbols=input_alphabet)
        codec = _TokenCodec()
        counts: Counter[EdgeKey] = Counter()
        states: list[_CompactEdgeTree] = []

        for graph in graphs:
            if self.validate_inputs:
                validate_coarsenable_tree(
                    graph,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                )
            states.append(
                _CompactEdgeTree.from_graph(
                    graph,
                    codec=codec,
                    vocab=vocab,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                    attach_attr=self.attach_attr,
                    pair_counts=counts,
                    capture_output=False,
                    build_edge_index=True,
                )
            )

        label_counts, label_sizes = _initial_label_statistics(states, codec, vocab)

        learned: list[EdgeStarRule] = []
        encoding_rules: list[EncodingRule] = []
        self.history_ = []
        rank = 0
        while self.num_merges is None or rank < self.num_merges:
            best = self._select_best_pair(counts, codec, label_counts, label_sizes)
            if best is None:
                break
            key = best.key
            parent_id, child_id = key
            parent_label = codec.decode(parent_id)
            child_label = codec.decode(child_id)

            # One fitting label covers every contracted group of this rank,
            # regardless of how many children each group absorbs.  Its nominal
            # size (used only for size-weighted scoring of later ranks) is the
            # single-child size; actual per-occurrence sizes are tracked exactly
            # on the compact tree and in each occurrence's structural type.
            token = edge_star_token(rank)
            new_id = codec.intern(token)
            nominal_size = label_sizes[parent_id] + label_sizes[child_id]
            _set_new_label_statistics(
                label_counts, label_sizes, label_id=new_id, size=nominal_size
            )

            total_events = 0
            total_children = 0
            for state in states:
                group_count, child_count = state.contract_star_pair(
                    key, new_label=new_id, rule_token=token, pair_counts=counts
                )
                total_events += group_count
                total_children += child_count

            if total_events == 0:
                # With correct incremental counts this cannot happen: any
                # nonempty matching-edge set has at least one contractible group.
                counts.pop(key, None)
                continue

            # Update incremental label occurrence counts.  Each contracted group
            # consumes one parent occurrence and its children, and produces one
            # occurrence of the new token.
            if parent_id == child_id:
                label_counts[parent_id] -= total_events + total_children
            else:
                label_counts[parent_id] -= total_events
                label_counts[child_id] -= total_children
            label_counts[new_id] += total_events
            if label_counts[parent_id] < 0 or label_counts[child_id] < 0:
                raise RuntimeError("incremental label occurrence count became negative.")

            rule = EdgeStarRule(
                rank=rank,
                token=token,
                parent_label=parent_label,
                child_label=child_label,
                count=best.count,
                score=best.score,
                parent_count=best.parent_count,
                child_count=best.child_count,
                parent_size=best.parent_size,
                child_size=best.child_size,
            )
            learned.append(rule)
            encoding_rules.append(
                EncodingRule(
                    token=token,
                    operation="siblings",
                    created_at_step=rank,
                    pattern={
                        "parent_label": parent_label,
                        "child_label": child_label,
                    },
                    score=best.score,
                    metadata={
                        "actual_events": total_events,
                        "children_absorbed": total_children,
                        "count_semantics": "raw_matching_edges",
                        "pair_score": self.pair_score_display_name_,
                        "raw_count": best.count,
                        "parent_count": best.parent_count,
                        "child_count": best.child_count,
                        "parent_size": best.parent_size,
                        "child_size": best.child_size,
                    },
                )
            )
            self.history_.append(
                {
                    "rank": rank,
                    "token": token,
                    "parent_label": parent_label,
                    "child_label": child_label,
                    # Compatibility aliases.
                    "parent_token": parent_label,
                    "child_token": child_label,
                    "count": best.count,
                    "count_semantics": "raw_matching_edges",
                    "parent_count": best.parent_count,
                    "child_count": best.child_count,
                    "parent_size": best.parent_size,
                    "child_size": best.child_size,
                    "score": best.score,
                    "pair_score": self.pair_score_display_name_,
                    "actual_events": total_events,
                    "children_absorbed": total_children,
                }
            )
            rank += 1

        output_raw = all(graph.graph.get(RAW_INPUT_FLAG, False) for graph in graphs)
        encoder = EdgeBPEWithAutoStarEncoder(
            model_id=self.model_id,
            vocab=vocab,
            rules=tuple(encoding_rules),
            base_labels=frozenset(input_alphabet),
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_label_attr=self.super_label_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            edge_rules=tuple(learned),
        )
        decoder = StructuralStageDecoder(
            model_id=self.model_id,
            vocab=vocab,
            base_labels=frozenset(input_alphabet),
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_label_attr=self.super_label_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            output_raw=output_raw,
        )
        return encoder, decoder

    def _select_best_pair(
        self,
        counts: Counter[EdgeKey],
        codec: _TokenCodec,
        label_counts: Sequence[int],
        label_sizes: Sequence[int],
    ) -> _PairSelection | None:
        best: _PairSelection | None = None
        best_priority: tuple[float, int, str, str] | None = None
        for key, count in counts.items():
            if count < self.min_pair_count:
                continue
            parent_id, child_id = key
            parent_count = int(label_counts[parent_id])
            child_count = int(label_counts[child_id])
            parent_size = int(label_sizes[parent_id])
            child_size = int(label_sizes[child_id])
            try:
                score = float(
                    self._pair_score_function(
                        int(count),
                        parent_count,
                        child_count,
                        parent_size,
                        child_size,
                    )
                )
            except Exception as exc:
                raise ValidationError(
                    f"pair_score failed for pair "
                    f"({codec.decode(parent_id)!r}, {codec.decode(child_id)!r})."
                ) from exc
            if not math.isfinite(score):
                raise ValidationError(
                    f"pair_score returned non-finite value {score!r} for "
                    f"N(A,B)={count}, N(A)={parent_count}, N(B)={child_count}, "
                    f"S(A)={parent_size}, S(B)={child_size}."
                )
            priority = (
                score,
                count,
                codec.sort_key(parent_id),
                codec.sort_key(child_id),
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best = _PairSelection(
                    key=key,
                    count=int(count),
                    parent_count=parent_count,
                    child_count=child_count,
                    parent_size=parent_size,
                    child_size=child_size,
                    score=score,
                )
        return best
