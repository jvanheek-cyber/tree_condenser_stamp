"""Edge-only BPE coarsener for directed labeled trees.

The public boundary uses NetworkX graphs and tuple-valued ``attach_map``
attributes.  Fitting and encoding use a compact mutable tree with integer token
ids and scalar attachment sites.  The scalar representation is exact here
because base tokens and edge-only BPE tokens always expose exactly one root.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable as HashableABC, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import StagedTreeDecoder, TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..nx_io import edge_attach_attrs
from ..provenance import NODE_ATTRS_KEY, PROVENANCE_KEY
from ..validation import validate_encoded_tree, validate_raw_tree
from ..vocabulary import AttachMap, Token, VocabEntry, Vocabulary, base_token

# Edge-only tokens expose one root, so a live edge needs one integer attachment
# site internally.  The public API converts it back to ``(attach_site,)``.
EdgeKey = tuple[int, int, int]


def edge_bpe_token(rank: int) -> tuple[str, int]:
    """Stable token id for the ``rank``-th learned edge-BPE merge."""

    if rank < 0:
        raise ValidationError("edge-BPE rank must be nonnegative.")
    return ("edge_bpe", int(rank))


@dataclass(frozen=True, slots=True)
class EdgeBPERule:
    """One fitted edge-contraction rule.

    ``count`` is the raw number of matching encoded edges when the rule was
    selected.  Overlapping occurrences are included, even though the rule's
    contraction pass can merge only a vertex-disjoint subset of them.
    """

    rank: int
    token: Token
    parent_token: Token
    child_token: Token
    attach_map: AttachMap
    count: int

    def __post_init__(self) -> None:
        if len(self.attach_map) != 1:
            raise ValidationError("edge-only BPE rules require a one-entry attach_map.")
        site = self.attach_map[0]
        if not isinstance(site, int) or isinstance(site, bool) or site < 0:
            raise ValidationError(f"invalid edge attachment site {site!r}.")

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

    Repeated edge contractions create one pair of integer references instead of
    repeatedly copying growing UID tuples.  Flattening occurs once, when the
    final encoded NetworkX graph is produced.
    """

    leaves: list[Any]
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)

    def merge(self, left_ref: int, right_ref: int) -> int:
        ref = len(self.leaves) + len(self.left)
        self.left.append(left_ref)
        self.right.append(right_ref)
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
            if merge_index < 0 or merge_index >= len(self.left):
                raise RuntimeError(f"invalid provenance-rope reference {current!r}.")
            # Stack is LIFO; push right first to preserve left-to-right site order.
            stack.append(self.right[merge_index])
            stack.append(self.left[merge_index])
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
    """Mutable array-backed tree used by the edge-BPE inner loop.

    The central arrays are ``parent``, ``children``, ``label``, and ``time``.
    Contraction keeps the parent position alive and removes the child position,
    so array lengths never grow.  Fit-time states omit provenance and all
    output-only metadata.
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

    def contract_and_count_pairs(
        self,
        key: EdgeKey,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> int:
        """Contract a deterministic vertex-disjoint subset of matching edges.

        ``pair_counts`` continues to count all live matching edges, including
        overlapping ones.  Local updates remove the edges incident to each
        contracted pair, mutate the compact tree, and add the replacement
        incident edges.
        """

        bucket = self.edge_index.get(key)
        if not bucket:
            return 0
        candidates = sorted(bucket, key=self._edge_sort_key)
        used: set[int] = set()
        n_events = 0

        for child in candidates:
            if not self._edge_is_live(child):
                continue
            parent = self.parent[child]
            if parent in used or child in used:
                continue
            if self._edge_key(child) != key:
                continue
            self._contract_edge(parent, child, new_label=new_label, pair_counts=pair_counts)
            used.add(parent)
            used.add(child)
            n_events += 1
        return n_events

    def _contract_edge(
        self,
        parent_node: int,
        child_node: int,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None,
    ) -> int:
        """Contract ``parent_node -> child_node`` in place at ``parent_node``."""

        if not self._edge_is_live(child_node) or self.parent[child_node] != parent_node:
            raise ValidationError("attempted to contract a non-live edge occurrence.")

        grandparent = self.parent[parent_node]
        parent_site_count = self._site_count(self.label[parent_node])
        old_parent_children = self.children[parent_node]
        child_children = self.children[child_node]

        # Remove every edge whose key will disappear or change.  This is the
        # incoming edge to the surviving parent, all old parent edges (including
        # the contracted edge), and all outgoing edges of the removed child.
        if grandparent != -1:
            self._remove_edge_from_index(parent_node, pair_counts=pair_counts)

        remaining_parent_children: list[int] = []
        found_child = False
        for current_child in old_parent_children:
            self._remove_edge_from_index(current_child, pair_counts=pair_counts)
            if current_child == child_node:
                found_child = True
            else:
                remaining_parent_children.append(current_child)
        if not found_child:
            raise RuntimeError("contracted child is missing from its parent's child list.")

        for current_child in child_children:
            self._remove_edge_from_index(current_child, pair_counts=pair_counts)

        # The parent array position survives.  This avoids appending replacement
        # nodes and keeps compact arrays bounded by the original number of nodes.
        self.label[parent_node] = new_label
        self.time[parent_node] = min(self.time[parent_node], self.time[child_node])

        if self.uid_ref is not None:
            if self.output is None:
                raise RuntimeError("uid_ref exists without an output context.")
            self.uid_ref[parent_node] = self.output.uid_rope.merge(
                self.uid_ref[parent_node], self.uid_ref[child_node]
            )
            self.uid_ref[child_node] = -1

        for current_child in child_children:
            self.parent[current_child] = parent_node
            self.attach_to_parent[current_child] += parent_site_count

        remaining_parent_children.extend(child_children)
        self.children[parent_node] = remaining_parent_children

        self.alive[child_node] = False
        self.parent[child_node] = -1
        self.children[child_node] = []
        self.attach_to_parent[child_node] = -1

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


@dataclass
class EdgeBPEEncoder(TreeEncoder):
    """Encoder artifact for :class:`EdgeBPECoarsener`."""

    edge_rules: tuple[EdgeBPERule, ...] = ()

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
            new_id = state.codec.intern(rule.token)
            state.contract_and_count_pairs(
                (parent_id, child_id, rule.attach_site),
                new_label=new_id,
                pair_counts=None,
            )

        return state.to_networkx(validate=validate)


class EdgeBPECoarsener(TreeCoarsener):
    """Byte-pair-style coarsener that learns only edge contractions.

    At each step, the score of a pair is its raw number of matching encoded
    edges ``(parent_token, child_token, attachment_site)``.  Overlapping matches
    count toward the score.  After a rule is selected, its transform contracts a
    deterministic vertex-disjoint subset of those occurrences, exactly as a BPE
    pass must avoid using one token occurrence twice.
    """

    def __init__(
        self,
        *,
        num_merges: int | None = None,
        min_pair_count: int = 2,
        backend: Literal["python", "numba"] = "python",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_merges is not None and num_merges < 0:
            raise ValueError("num_merges must be None or nonnegative.")
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be at least 1.")
        if backend not in {"python", "numba"}:
            raise ValueError("backend must be 'python' or 'numba'.")
        self.num_merges = num_merges
        self.min_pair_count = min_pair_count
        self.backend = backend
        self.history_: list[dict[str, Any]] = []

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        use_numba = self.backend == "numba"
        numba_forest: Any | None = None
        if use_numba:
            from .edge_bpe_numba import NumbaTrainingForest, require_numba

            require_numba()

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
                    pair_counts=None if use_numba else pair_counts,
                    base_labels=base_labels,
                    capture_output=False,
                    build_edge_index=not use_numba,
                )
            )

        if use_numba:
            max_possible_merges = sum(len(state.parent) - 1 for state in states)
            if self.num_merges is not None:
                max_possible_merges = min(max_possible_merges, self.num_merges)
            numba_forest = NumbaTrainingForest.from_compact_states(
                states,
                label_capacity=len(codec.id_to_token) + max_possible_merges,
            )
            # The compiled forest now owns independent NumPy arrays; release the
            # temporary Python list-of-lists states before entering the merge loop.
            states.clear()

        edge_rules: list[EdgeBPERule] = []
        encoding_rules: list[EncodingRule] = []
        self.history_ = []

        rank = 0
        while self.num_merges is None or rank < self.num_merges:
            if numba_forest is None:
                best = self._select_best_pair(pair_counts, codec)
            else:
                best = numba_forest.select_best_pair(self.min_pair_count, codec)
            if best is None:
                break
            best_key, best_count = best
            parent_id, child_id, attach_site = best_key
            parent_token = codec.decode(parent_id)
            child_token = codec.decode(child_id)
            token = edge_bpe_token(rank)
            public_attach_map = (attach_site,)

            entry = VocabEntry(
                token=token,
                parent=(-1, 0),
                label=(parent_token, child_token),
                attach=public_attach_map,
                created_at_step=rank,
                operation="edge",
                score=float(best_count),
                metadata={
                    "coarsener": "EdgeBPECoarsener",
                    "rank": rank,
                    "count": best_count,
                    "count_semantics": "raw_matching_edges",
                    "min_pair_count": self.min_pair_count,
                    "parent_token": parent_token,
                    "child_token": child_token,
                    "attach_map": public_attach_map,
                },
            )
            vocab.add(entry)
            new_id = codec.intern(token)

            if numba_forest is None:
                actual_events = 0
                for state in states:
                    actual_events += state.contract_and_count_pairs(
                        best_key,
                        new_label=new_id,
                        pair_counts=pair_counts,
                    )
            else:
                numba_forest.register_label(new_id, vocab.site_count(token))
                actual_events = numba_forest.contract_pair(best_key, new_label=new_id)

            if actual_events == 0:
                # With correct incremental counts this cannot happen: any
                # nonempty matching-edge set has at least one contractible edge.
                # Keep a defensive rollback for the Python index.  The compiled
                # index deliberately exposes no mutation hook for an impossible
                # stale bucket, so fail loudly rather than looping forever.
                if numba_forest is None:
                    pair_counts.pop(best_key, None)
                    vocab.remove_last(token)
                    continue
                raise RuntimeError(
                    "Numba pair index selected a positive-count pair with no "
                    "contractible occurrence."
                )

            rule = EdgeBPERule(
                rank=rank,
                token=token,
                parent_token=parent_token,
                child_token=child_token,
                attach_map=public_attach_map,
                count=best_count,
            )
            edge_rules.append(rule)
            encoding_rules.append(
                EncodingRule(
                    token=token,
                    operation="edge",
                    created_at_step=rank,
                    pattern={
                        "parent_token": parent_token,
                        "child_token": child_token,
                        "attach_map": public_attach_map,
                    },
                    score=float(best_count),
                    metadata={
                        "actual_events": actual_events,
                        "count_semantics": "raw_matching_edges",
                    },
                )
            )
            self.history_.append(
                {
                    "rank": rank,
                    "token": token,
                    "parent_token": parent_token,
                    "child_token": child_token,
                    "attach_map": public_attach_map,
                    "count": best_count,
                    "count_semantics": "raw_matching_edges",
                    "actual_events": actual_events,
                }
            )
            rank += 1

        encoder = EdgeBPEEncoder(
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
