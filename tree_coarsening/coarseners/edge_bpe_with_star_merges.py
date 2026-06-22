"""Edge-BPE coarsener with star merges for directed labeled trees.

This variant behaves like :mod:`edge_bpe` when selecting which pair to merge:
at each step the highest-scoring encoded edge
``(parent_token, child_token, attachment_site)`` is chosen by its raw matching
edge count.  It differs in the *transform* step.  Instead of contracting a
single parent/child pair, every parent that has several matching children at the
same attachment site has *all* of those children contracted into the parent at
once, exactly as :class:`~tree_coarsening.coarseners.star.StarCoarsener` would
contract a labeled starburst -- but restricted to the candidate children of the
selected pair rather than to every star in the tree.

A merge that contracts ``k`` matching children produces an arity-``k`` token.
As with the star coarsener, distinct arities create distinct vocabulary tokens,
so a single selected pair can introduce one token per observed arity.

The public boundary uses NetworkX graphs and tuple-valued ``attach_map``
attributes.  Fitting and encoding use a compact mutable tree with integer token
ids and scalar attachment sites.  The scalar representation is exact here
because base tokens and the tokens produced by this coarsener always expose
exactly one root.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable as HashableABC, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import StagedTreeDecoder, TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..nx_io import edge_attach_attrs
from ..provenance import NODE_ATTRS_KEY, PROVENANCE_KEY
from ..validation import validate_encoded_tree, validate_raw_tree
from ..vocabulary import AttachMap, Token, VocabEntry, Vocabulary, base_token

# The tokens produced here expose one root, so a live edge needs one integer
# attachment site internally.  The public API converts it back to
# ``(attach_site,)``.
EdgeKey = tuple[int, int, int]

# Maps an arity to ``(token, interned_id, new_site_count)`` for that arity, or
# ``None`` when no token exists for the arity (used by encode time).
TokenForArity = Callable[[int], "tuple[Token, int, int] | None"]


def edge_bpe_token(rank: int) -> tuple[str, int]:
    """Stable token id for the arity-one (single child) merge at ``rank``."""

    if rank < 0:
        raise ValidationError("edge-BPE rank must be nonnegative.")
    return ("edge_bpe", int(rank))


def edge_star_token(rank: int, arity: int) -> tuple[str, int, int]:
    """Stable token id for the ``rank``-th merge that contracts ``arity`` children.

    Arity one reuses :func:`edge_bpe_token` so a star variant that never finds a
    multi-child group is byte-for-byte compatible with the plain edge coarsener.
    """

    if rank < 0:
        raise ValidationError("edge-star rank must be nonnegative.")
    if arity < 1:
        raise ValidationError("edge-star arity must be positive.")
    return ("edge_star", int(rank), int(arity))


def merge_token(rank: int, arity: int) -> Token:
    """Return the token id for a merge of ``arity`` children at ``rank``."""

    if arity == 1:
        return edge_bpe_token(rank)
    return edge_star_token(rank, arity)


@dataclass(frozen=True, slots=True)
class EdgeStarRule:
    """One fitted merge rule for a selected ``(parent, child, site)`` pair.

    ``count`` is the raw number of matching encoded edges when the rule was
    selected.  ``arities`` lists every contracted-group size observed when the
    rule's transform ran, and ``tokens`` is the position-aligned token id for
    each arity.  Each arity maps to its own vocabulary entry.
    """

    rank: int
    parent_token: Token
    child_token: Token
    attach_map: AttachMap
    count: int
    arities: tuple[int, ...]
    tokens: tuple[Token, ...]

    def __post_init__(self) -> None:
        if len(self.attach_map) != 1:
            raise ValidationError("edge-only merge rules require a one-entry attach_map.")
        site = self.attach_map[0]
        if not isinstance(site, int) or isinstance(site, bool) or site < 0:
            raise ValidationError(f"invalid edge attachment site {site!r}.")
        if len(self.arities) != len(self.tokens):
            raise ValidationError("arities and tokens must be position aligned.")
        if any(arity < 1 for arity in self.arities):
            raise ValidationError("edge-star arities must be positive.")

    @property
    def attach_site(self) -> int:
        return self.attach_map[0]


@dataclass(slots=True)
class _TokenCodec:
    """Dense integer interner for token ids used by compact tree states."""

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
    """Append-only concatenation tree for occurrence provenance.

    Repeated contractions create one node per merge instead of repeatedly
    copying growing UID tuples.  Flattening occurs once, when the final encoded
    NetworkX graph is produced.  A merge may concatenate more than two children
    (a star group), so each rope node stores an ordered list of references.
    """

    leaves: list[Any]
    refs: list[tuple[int, ...]] = field(default_factory=list)

    def merge(self, parts: Sequence[int]) -> int:
        ref = len(self.leaves) + len(self.refs)
        self.refs.append(tuple(parts))
        return ref

    def flatten(self, ref: int) -> tuple[Any, ...]:
        leaf_count = len(self.leaves)
        out: list[Any] = []
        stack = [ref]
        while stack:
            current = stack.pop()
            if current < leaf_count:
                out.append(self.leaves[current])
                continue
            merge_index = current - leaf_count
            if merge_index < 0 or merge_index >= len(self.refs):
                raise RuntimeError(f"invalid provenance-rope reference {current!r}.")
            # Stack is LIFO; push in reverse to preserve left-to-right order.
            for part in reversed(self.refs[merge_index]):
                stack.append(part)
        return tuple(out)


@dataclass(slots=True)
class _OutputContext:
    """Data retained only when a compact state will be emitted as NetworkX."""

    label_attr: str
    super_uid_attr: str
    attach_attr: str
    model_id: str
    provenance: dict[str, Any]
    uid_rope: _UidRope


def _bump_count(counts: Counter[EdgeKey], key: EdgeKey, delta: int) -> None:
    """Increment or decrement a pair count, deleting zero entries."""

    new_value = counts.get(key, 0) + delta
    if new_value < 0:
        raise RuntimeError(f"edge-pair count for {key!r} became negative.")
    if new_value == 0:
        counts.pop(key, None)
    else:
        counts[key] = new_value


@dataclass(slots=True)
class _CompactEdgeTree:
    """Mutable array-backed tree used by the edge-star inner loop.

    The central arrays are ``parent``, ``children``, ``label``, and ``time``.
    Contraction keeps the parent position alive and removes the contracted
    children positions, so array lengths never grow.  Fit-time states omit
    provenance and all output-only metadata.
    """

    parent: list[int]
    children: list[list[int]]
    label: list[int]
    time: list[float]
    attach_to_parent: list[int]
    alive: list[bool]
    codec: _TokenCodec
    vocab: Vocabulary
    edge_index: dict[EdgeKey, set[int]] = field(default_factory=lambda: defaultdict(set))
    output: _OutputContext | None = None
    uid_ref: list[int] | None = None

    @classmethod
    def from_raw_graph(
        cls,
        G: nx.DiGraph,
        *,
        codec: _TokenCodec,
        vocab: Vocabulary,
        label_attr: str = "label",
        time_attr: str = "time",
        uid_attr: str = "uid",
        super_uid_attr: str = "super_uids",
        attach_attr: str = "attach_map",
        model_id: str = "",
        pair_counts: Counter[EdgeKey] | None = None,
        base_labels: set[str] | None = None,
        capture_output: bool = False,
        build_edge_index: bool = True,
    ) -> "_CompactEdgeTree":
        """Build a compact state without copying the input NetworkX graph."""

        roots = [node for node in G.nodes if G.in_degree(node) == 0]
        if len(roots) != 1:
            raise ValidationError(f"expected exactly one root; found {len(roots)}.")

        root = roots[0]
        order: list[Any] = [root]
        seen: set[Any] = {root}
        parent = [-1]
        children: list[list[int]] = [[]]

        def child_sort_key(child: Any) -> tuple[Any, ...]:
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
            ordered_children = sorted(G.successors(node), key=child_sort_key)
            for child in ordered_children:
                if child in seen:
                    raise ValidationError(
                        f"node {child!r} is reachable more than once; input is not a tree."
                    )
                child_index = len(order)
                seen.add(child)
                order.append(child)
                parent.append(cursor)
                children.append([])
                children[cursor].append(child_index)
            cursor += 1

        if len(order) != G.number_of_nodes():
            raise ValidationError("not every node is reachable from the directed root.")

        labels: list[int] = []
        times: list[float] = []
        for node in order:
            data = G.nodes[node]
            raw_label = data[label_attr]
            if base_labels is not None:
                base_labels.add(raw_label)
            labels.append(codec.intern(base_token(raw_label)))
            times.append(float(data[time_attr]))

        n = len(order)
        attach_to_parent = [-1] + [0] * (n - 1)
        alive = [True] * n
        output: _OutputContext | None = None
        uid_ref: list[int] | None = None

        if capture_output:
            uids: list[Any] = []
            attrs_by_uid: dict[Any, dict[str, Any]] = {}
            for node in order:
                data = G.nodes[node]
                uid = data.get(uid_attr, node)
                if not isinstance(uid, HashableABC):
                    raise ValidationError(f"uid for node {node!r} must be hashable; got {uid!r}.")
                if uid in attrs_by_uid:
                    raise ValidationError(f"duplicate uid after fallback assignment: {uid!r}.")
                attrs = dict(data)
                attrs[uid_attr] = uid
                attrs_by_uid[uid] = attrs
                uids.append(uid)

            uid_rope = _UidRope(leaves=uids)
            uid_ref = list(range(n))
            output = _OutputContext(
                label_attr=label_attr,
                super_uid_attr=super_uid_attr,
                attach_attr=attach_attr,
                model_id=model_id,
                provenance={NODE_ATTRS_KEY: attrs_by_uid, "uid_attr": uid_attr},
                uid_rope=uid_rope,
            )

        state = cls(
            parent=parent,
            children=children,
            label=labels,
            time=times,
            attach_to_parent=attach_to_parent,
            alive=alive,
            codec=codec,
            vocab=vocab,
            output=output,
            uid_ref=uid_ref,
        )
        if build_edge_index:
            state.rebuild_edge_index(pair_counts=pair_counts)
        return state

    def rebuild_edge_index(self, *, pair_counts: Counter[EdgeKey] | None = None) -> None:
        self.edge_index = defaultdict(set)
        for child in range(len(self.parent)):
            if self._edge_is_live(child):
                self._add_edge_to_index(child, pair_counts=pair_counts)

    def _edge_is_live(self, child: int) -> bool:
        if child < 0 or child >= len(self.alive) or not self.alive[child]:
            return False
        p = self.parent[child]
        return p >= 0 and p < len(self.alive) and self.alive[p]

    def _edge_key_unchecked(self, child: int) -> EdgeKey:
        p = self.parent[child]
        return (self.label[p], self.label[child], self.attach_to_parent[child])

    def _edge_key(self, child: int) -> EdgeKey:
        if not self._edge_is_live(child):
            raise ValidationError(f"node {child!r} does not have a live incoming edge.")
        return self._edge_key_unchecked(child)

    def _add_edge_to_index(
        self,
        child: int,
        *,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> None:
        key = self._edge_key_unchecked(child)
        self.edge_index[key].add(child)
        if pair_counts is not None:
            _bump_count(pair_counts, key, +1)

    def _remove_edge_from_index(
        self,
        child: int,
        *,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> None:
        key = self._edge_key_unchecked(child)
        bucket = self.edge_index.get(key)
        if bucket is None or child not in bucket:
            raise RuntimeError(f"live edge for child {child!r} is missing from its index bucket.")
        bucket.remove(child)
        if not bucket:
            self.edge_index.pop(key, None)
        if pair_counts is not None:
            _bump_count(pair_counts, key, -1)

    def _site_count(self, label_id: int) -> int:
        # Vocabulary counts are cached, so this is an O(1) lookup after token decode.
        return self.vocab.site_count(self.codec.decode(label_id))

    def _edge_sort_key(self, child: int) -> tuple[float, float, int, int]:
        p = self.parent[child]
        return (self.time[child], self.time[p], p, child)

    def contract_star_groups(
        self,
        key: EdgeKey,
        *,
        token_for_arity: TokenForArity,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> dict[int, int]:
        """Contract matching children into their parents as star groups.

        Every parent that has one or more live children matching ``key`` has all
        of those children contracted into it at once.  The number of merged
        children (the arity) selects the replacement token via
        ``token_for_arity``; an arity whose token is missing (``None``) is left
        uncontracted.  Returns a mapping from arity to the number of merge
        events of that arity.

        ``pair_counts`` continues to count all live matching edges, including
        overlapping ones from other parents.  Local updates remove the edges
        incident to each contracted group, mutate the compact tree, and add the
        replacement incident edges.
        """

        bucket = self.edge_index.get(key)
        if not bucket:
            return {}

        # Group candidate children by their (live) parent.  Children are sorted
        # deterministically so the contracted recipe order is reproducible.
        groups: dict[int, list[int]] = defaultdict(list)
        for child in sorted(bucket, key=self._edge_sort_key):
            if not self._edge_is_live(child):
                continue
            if self._edge_key_unchecked(child) != key:
                continue
            groups[self.parent[child]].append(child)

        events_by_arity: dict[int, int] = defaultdict(int)
        # Sort parents for a deterministic contraction order.  A parent is never
        # itself one of the matching children of another group for the same key
        # (that would require two live incoming edges), so groups are disjoint.
        for parent_node in sorted(groups, key=lambda p: (self.time[p], p)):
            members = groups[parent_node]
            arity = len(members)
            resolved = token_for_arity(arity)
            if resolved is None:
                continue
            _token, new_label, new_site_count = resolved
            self._contract_star_group(
                parent_node,
                members,
                new_label=new_label,
                new_site_count=new_site_count,
                pair_counts=pair_counts,
            )
            events_by_arity[arity] += 1
        return dict(events_by_arity)

    def _contract_star_group(
        self,
        parent_node: int,
        child_nodes: Sequence[int],
        *,
        new_label: int,
        new_site_count: int,
        pair_counts: Counter[EdgeKey] | None,
    ) -> int:
        """Contract ``parent_node`` together with all ``child_nodes`` in place.

        The new token's site layout is the parent's sites followed by each
        contracted child's sites in ``child_nodes`` order.  Grandchildren of the
        ``j``-th contracted child shift by the parent's site count plus the
        cumulative site count of the children that precede it.
        """

        if not child_nodes:
            raise ValidationError("a star group must contain at least one child.")

        for child_node in child_nodes:
            if not self._edge_is_live(child_node) or self.parent[child_node] != parent_node:
                raise ValidationError("attempted to contract a non-live edge occurrence.")

        grandparent = self.parent[parent_node]
        parent_site_count = self._site_count(self.label[parent_node])
        child_set = set(child_nodes)

        # Remove every edge whose key will disappear or change: the parent's
        # incoming edge, every old parent edge (the contracted children plus the
        # parent's other children), and the outgoing edges of each contracted
        # child.
        if grandparent != -1:
            self._remove_edge_from_index(parent_node, pair_counts=pair_counts)

        retained_parent_children: list[int] = []
        for current_child in self.children[parent_node]:
            self._remove_edge_from_index(current_child, pair_counts=pair_counts)
            if current_child not in child_set:
                retained_parent_children.append(current_child)

        for child_node in child_nodes:
            for grandchild in self.children[child_node]:
                self._remove_edge_from_index(grandchild, pair_counts=pair_counts)

        # The parent array position survives.  This avoids appending replacement
        # nodes and keeps compact arrays bounded by the original number of nodes.
        merged_time = self.time[parent_node]
        for child_node in child_nodes:
            merged_time = min(merged_time, self.time[child_node])
        self.label[parent_node] = new_label
        self.time[parent_node] = merged_time

        if self.uid_ref is not None:
            if self.output is None:
                raise RuntimeError("uid_ref exists without an output context.")
            parts = [self.uid_ref[parent_node]]
            parts.extend(self.uid_ref[child_node] for child_node in child_nodes)
            self.uid_ref[parent_node] = self.output.uid_rope.merge(parts)

        # Reparent grandchildren, offsetting their attachment site into the new
        # token's coordinate space.  The offset advances by each child's site
        # count so the layout matches the recipe (P, C_1, ..., C_k).
        new_parent_children = retained_parent_children
        site_offset = parent_site_count
        for child_node in child_nodes:
            child_site_count = self._site_count(self.label[child_node])
            for grandchild in self.children[child_node]:
                self.parent[grandchild] = parent_node
                self.attach_to_parent[grandchild] += site_offset
                new_parent_children.append(grandchild)
            site_offset += child_site_count

        if site_offset != new_site_count:
            raise RuntimeError(
                f"star merge site layout mismatch: computed {site_offset}, "
                f"vocabulary reports {new_site_count}."
            )

        for child_node in child_nodes:
            self.alive[child_node] = False
            self.parent[child_node] = -1
            self.children[child_node] = []
            self.attach_to_parent[child_node] = -1
            if self.uid_ref is not None:
                self.uid_ref[child_node] = -1

        self.children[parent_node] = new_parent_children

        # Reinsert the changed incoming edge and all outgoing edges of the new
        # token occurrence.  Counts remain raw edge-occurrence counts.
        if grandparent != -1:
            self._add_edge_to_index(parent_node, pair_counts=pair_counts)
        for current_child in self.children[parent_node]:
            self._add_edge_to_index(current_child, pair_counts=pair_counts)

        return parent_node

    def to_networkx(self, *, validate: bool = True) -> nx.DiGraph:
        """Emit the minimal encoded-graph schema from an output-enabled state."""

        if self.output is None or self.uid_ref is None:
            raise RuntimeError("fit-time compact states cannot be converted to NetworkX output.")

        context = self.output
        live_nodes = [node for node, is_alive in enumerate(self.alive) if is_alive]
        # Initial ids are topological and contractions retain the ancestor's id,
        # so increasing live ids remain a valid topological order.
        mapping = {old_node: new_node for new_node, old_node in enumerate(live_nodes)}

        H = nx.DiGraph()
        H.graph[PROVENANCE_KEY] = context.provenance
        H.graph["tree_coarsening_schema"] = {
            "schema_version": "0.2",
            "model_id": context.model_id,
            "node_label_semantics": "encoded token id",
            "super_uid_attr": context.super_uid_attr,
            "attach_attr": context.attach_attr,
        }

        for old_node in live_nodes:
            H.add_node(
                mapping[old_node],
                **{
                    context.label_attr: self.codec.decode(self.label[old_node]),
                    context.super_uid_attr: context.uid_rope.flatten(self.uid_ref[old_node]),
                },
            )

        for old_child in live_nodes:
            old_parent = self.parent[old_child]
            if old_parent == -1:
                continue
            if not self.alive[old_parent]:
                raise ValidationError(
                    f"live node {old_child!r} has dead parent {old_parent!r}."
                )
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
                super_uid_attr=context.super_uid_attr,
                attach_attr=context.attach_attr,
            )
        return H


def _star_vocab_entry(
    token: Token,
    *,
    parent_token: Token,
    child_token: Token,
    attach_site: int,
    arity: int,
    rank: int,
    count: int,
    min_pair_count: int,
) -> VocabEntry:
    """Build the recipe for a merge of ``arity`` ``child_token`` children.

    The recipe is the parent token followed by ``arity`` copies of the child
    token, every child attaching to the parent (component 0) at ``attach_site``.
    """

    public_attach_map = (attach_site,)
    label = (parent_token, *([child_token] * arity))
    parent = (-1, *([0] * arity))
    attach = tuple(attach_site for _ in range(arity))
    return VocabEntry(
        token=token,
        parent=parent,
        label=label,
        attach=attach,
        created_at_step=rank,
        operation="edge" if arity == 1 else "siblings",
        score=float(count),
        metadata={
            "coarsener": "EdgeBPEWithStarMergesCoarsener",
            "rank": rank,
            "arity": arity,
            "count": count,
            "count_semantics": "raw_matching_edges",
            "min_pair_count": min_pair_count,
            "parent_token": parent_token,
            "child_token": child_token,
            "attach_map": public_attach_map,
        },
    )


@dataclass
class EdgeBPEWithStarMergesEncoder(TreeEncoder):
    """Encoder artifact for :class:`EdgeBPEWithStarMergesCoarsener`."""

    edge_rules: tuple[EdgeStarRule, ...] = ()

    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        if validate:
            validate_raw_tree(
                G,
                label_attr=self.label_attr,
                time_attr=self.time_attr,
                uid_attr=self.uid_attr,
                require_uid=False,
            )

        codec = _TokenCodec()
        for label in sorted(self.base_labels):
            codec.intern(base_token(label))
        for token in self.vocab.creation_order:
            entry = self.vocab.entries[token]
            for child_token in entry.label:
                codec.intern(child_token)
            codec.intern(token)

        state = _CompactEdgeTree.from_raw_graph(
            G,
            codec=codec,
            vocab=self.vocab,
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            model_id=self.model_id,
            capture_output=True,
        )

        for rule in self.edge_rules:
            parent_id = state.codec.intern(rule.parent_token)
            child_id = state.codec.intern(rule.child_token)
            # Only arities present in the fitted vocabulary can contract; a
            # transform-time arity not seen during fit is left uncontracted,
            # matching the star coarsener's fixed-vocabulary behavior.
            token_by_arity: dict[int, Token] = dict(zip(rule.arities, rule.tokens))

            def token_for_arity(arity: int, _by=token_by_arity) -> tuple[Token, int, int] | None:
                token = _by.get(arity)
                if token is None:
                    return None
                token_id = state.codec.intern(token)
                return token, token_id, self.vocab.site_count(token)

            state.contract_star_groups(
                (parent_id, child_id, rule.attach_site),
                token_for_arity=token_for_arity,
                pair_counts=None,
            )

        return state.to_networkx(validate=validate)


class EdgeBPEWithStarMergesCoarsener(TreeCoarsener):
    """Byte-pair-style coarsener that contracts candidate children as stars.

    Pair selection is identical to :class:`EdgeBPECoarsener`: at each step the
    score of a pair is its raw number of matching encoded edges
    ``(parent_token, child_token, attachment_site)``, and overlapping matches
    count toward the score.  The transform differs.  After a rule is selected,
    every parent that has multiple matching children at the selected attachment
    site has *all* of those children contracted together with the parent into a
    single new node, exactly as the star coarsener would contract a labeled
    starburst -- but only over the candidate children of the selected pair.

    Distinct arities produce distinct vocabulary tokens, so a single selected
    pair may introduce one token per observed arity.
    """

    def __init__(
        self,
        *,
        num_merges: int | None = None,
        min_pair_count: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must be None or nonnegative.")
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be at least 1.")
        self.num_merges = num_merges
        self.min_pair_count = min_pair_count
        self.history_: list[dict[str, Any]] = []

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        vocab = Vocabulary()
        codec = _TokenCodec()
        pair_counts: Counter[EdgeKey] = Counter()
        base_labels: set[str] = set()
        states: list[_CompactEdgeTree] = []

        for G in graphs:
            if self.validate_inputs:
                validate_raw_tree(
                    G,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    require_uid=False,
                )
            states.append(
                _CompactEdgeTree.from_raw_graph(
                    G,
                    codec=codec,
                    vocab=vocab,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    pair_counts=pair_counts,
                    base_labels=base_labels,
                    capture_output=False,
                    build_edge_index=True,
                )
            )

        edge_rules: list[EdgeStarRule] = []
        encoding_rules: list[EncodingRule] = []
        self.history_ = []

        rank = 0
        while self.num_merges is None or rank < self.num_merges:
            best = self._select_best_pair(pair_counts, codec)
            if best is None:
                break
            best_key, best_count = best
            parent_id, child_id, attach_site = best_key
            parent_token = codec.decode(parent_id)
            child_token = codec.decode(child_id)
            public_attach_map = (attach_site,)

            # Lazily create one vocabulary token per arity actually contracted,
            # so a single selected pair can spawn several arities.  Tokens and
            # their interned ids are cached for the duration of this rank.
            tokens_by_arity: dict[int, Token] = {}

            def token_for_arity(arity: int) -> tuple[Token, int, int] | None:
                token = tokens_by_arity.get(arity)
                if token is None:
                    token = merge_token(rank, arity)
                    entry = _star_vocab_entry(
                        token,
                        parent_token=parent_token,
                        child_token=child_token,
                        attach_site=attach_site,
                        arity=arity,
                        rank=rank,
                        count=best_count,
                        min_pair_count=self.min_pair_count,
                    )
                    vocab.add(entry)
                    tokens_by_arity[arity] = token
                token_id = codec.intern(token)
                return token, token_id, vocab.site_count(token)

            events_by_arity: dict[int, int] = defaultdict(int)
            for state in states:
                for arity, n_events in state.contract_star_groups(
                    best_key,
                    token_for_arity=token_for_arity,
                    pair_counts=pair_counts,
                ).items():
                    events_by_arity[arity] += n_events

            if not tokens_by_arity:
                # With correct incremental counts this cannot happen: any
                # nonempty matching-edge set has at least one contractible group.
                pair_counts.pop(best_key, None)
                continue

            arities = tuple(sorted(tokens_by_arity))
            tokens = tuple(tokens_by_arity[arity] for arity in arities)
            rule = EdgeStarRule(
                rank=rank,
                parent_token=parent_token,
                child_token=child_token,
                attach_map=public_attach_map,
                count=best_count,
                arities=arities,
                tokens=tokens,
            )
            edge_rules.append(rule)

            actual_events = sum(events_by_arity.values())
            events_summary = {arity: events_by_arity[arity] for arity in arities}
            for arity in arities:
                token = tokens_by_arity[arity]
                encoding_rules.append(
                    EncodingRule(
                        token=token,
                        operation="edge" if arity == 1 else "siblings",
                        created_at_step=rank,
                        pattern={
                            "parent_token": parent_token,
                            "child_token": child_token,
                            "attach_map": public_attach_map,
                            "arity": arity,
                        },
                        score=float(best_count),
                        metadata={
                            "actual_events": events_by_arity[arity],
                            "count_semantics": "raw_matching_edges",
                        },
                    )
                )
            self.history_.append(
                {
                    "rank": rank,
                    "parent_token": parent_token,
                    "child_token": child_token,
                    "attach_map": public_attach_map,
                    "count": best_count,
                    "count_semantics": "raw_matching_edges",
                    "arities": arities,
                    "tokens": tokens,
                    "actual_events": actual_events,
                    "events_by_arity": events_summary,
                }
            )
            rank += 1

        encoder = EdgeBPEWithStarMergesEncoder(
            model_id=self.model_id,
            vocab=vocab,
            rules=tuple(encoding_rules),
            base_labels=frozenset(base_labels),
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            edge_rules=tuple(edge_rules),
        )
        decoder = StagedTreeDecoder(
            model_id=self.model_id,
            vocab=vocab,
            base_labels=frozenset(base_labels),
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
        )
        return encoder, decoder

    def _select_best_pair(
        self,
        pair_counts: Counter[EdgeKey],
        codec: _TokenCodec,
    ) -> tuple[EdgeKey, int] | None:
        """Return the highest raw-count pair without allocating a candidate list.

        Cached token sort keys preserve the earlier deterministic tie policy
        without repeatedly formatting token objects inside the fit loop.
        """

        best_key: EdgeKey | None = None
        best_count = 0
        best_priority: tuple[int, int, str, str] | None = None
        for key, count in pair_counts.items():
            if count < self.min_pair_count:
                continue
            parent_id, child_id, attach_site = key
            priority = (
                count,
                -attach_site,
                codec.sort_key(parent_id),
                codec.sort_key(child_id),
            )
            if best_priority is None or priority > best_priority:
                best_key = key
                best_count = count
                best_priority = priority
        if best_key is None:
            return None
        return best_key, best_count
