"""Deterministic contraction of connected components selected by UID or label."""

from __future__ import annotations

from collections.abc import Collection, Hashable as HashableABC, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Hashable, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import StagedTreeDecoder, TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..nx_io import edge_attach_attrs, relabel_to_consecutive_topological
from ..provenance import PROVENANCE_KEY, provenance_from_raw_graph
from ..validation import (
    copy_with_uids,
    deterministic_node_order,
    validate_encoded_tree,
    validate_raw_tree,
)
from ..vocabulary import Token, VocabEntry, Vocabulary, base_token

SelectorKind = Literal["uid", "label"]
ComponentPolicy = Literal["all", "largest"]


def named_component_token(
    selector: SelectorKind,
    parent: tuple[int, ...],
    label: tuple[Token, ...],
    attach: tuple[int, ...],
) -> tuple[str, str, int, str]:
    """Return a stable token id for one canonical connected-component recipe.

    The digest keeps encoded node labels compact; the complete, collision-checked
    recipe remains authoritative in the vocabulary.
    """

    hasher = sha256()
    for values in (parent, label, attach):
        hasher.update(len(values).to_bytes(8, "big"))
        for value in values:
            item = repr(value).encode("utf-8")
            hasher.update(len(item).to_bytes(8, "big"))
            hasher.update(item)
    return ("named_component", selector, len(parent), hasher.hexdigest()[:20])


@dataclass
class NamedVertexEncoder(TreeEncoder):
    """Contract maximal connected components selected by raw UID or raw label.

    Vocabulary entries are registered lazily during ``encode`` because this
    coarsener learns no graph patterns during ``fit``. The matching decoder must
    share this encoder's mutable ``Vocabulary`` object.
    """

    selector: SelectorKind = "uid"
    selected_values: frozenset[Hashable] = frozenset()
    component_policy: ComponentPolicy = "all"

    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        if validate:
            validate_raw_tree(
                G,
                label_attr=self.label_attr,
                time_attr=self.time_attr,
                uid_attr=self.uid_attr,
                require_uid=False,
            )
        raw = copy_with_uids(G, uid_attr=self.uid_attr)

        selected = self._selected_nodes(raw)
        components = self._selected_components(raw, selected)
        components = [component for component in components if len(component) >= 2]
        if self.component_policy == "largest" and components:
            # ``max`` keeps the first component on ties; components are already
            # in deterministic root order.
            components = [max(components, key=len)]

        component_for_node: dict[Hashable, tuple[Hashable, int]] = {}
        component_info: dict[Hashable, tuple[Token, tuple[Hashable, ...]]] = {}

        for serial, component in enumerate(components):
            parent, labels, attach = self._component_recipe(raw, component)
            token = self._register_recipe(parent, labels, attach)
            coarse_node = ("__tc_named_component__", self.model_id, serial)
            component_info[coarse_node] = (token, component)
            for site_i, node in enumerate(component):
                if node in component_for_node:
                    raise ValidationError(
                        f"node {node!r} belongs to more than one selected component."
                    )
                component_for_node[node] = (coarse_node, site_i)

        H = nx.DiGraph()
        H.graph[PROVENANCE_KEY] = provenance_from_raw_graph(raw, uid_attr=self.uid_attr)
        H.graph["tree_coarsening_schema"] = {
            "schema_version": "0.3",
            "model_id": self.model_id,
            "node_label_semantics": "encoded token id",
            "super_uid_attr": self.super_uid_attr,
            "attach_attr": self.attach_attr,
        }

        owner: dict[Hashable, Hashable] = {}
        site_in_owner: dict[Hashable, int] = {}

        for node, data in raw.nodes(data=True):
            membership = component_for_node.get(node)
            if membership is not None:
                coarse_node, site_i = membership
                owner[node] = coarse_node
                site_in_owner[node] = site_i
                continue

            token = base_token(data[self.label_attr])
            owner[node] = node
            site_in_owner[node] = 0
            H.add_node(
                node,
                **{
                    self.label_attr: token,
                    self.super_uid_attr: (data[self.uid_attr],),
                },
            )

        for coarse_node, (token, component) in component_info.items():
            H.add_node(
                coarse_node,
                **{
                    self.label_attr: token,
                    self.super_uid_attr: tuple(
                        raw.nodes[node][self.uid_attr] for node in component
                    ),
                },
            )

        # Base tokens and connected-component tokens both expose exactly one
        # root. Hence every surviving coarse edge needs one parent-site index.
        for u, v in raw.edges:
            coarse_u = owner[u]
            coarse_v = owner[v]
            if coarse_u == coarse_v:
                continue
            H.add_edge(
                coarse_u,
                coarse_v,
                **edge_attach_attrs((site_in_owner[u],), attach_attr=self.attach_attr),
            )

        H = relabel_to_consecutive_topological(H)
        if validate:
            validate_encoded_tree(
                H,
                vocab=self.vocab,
                label_attr=self.label_attr,
                super_uid_attr=self.super_uid_attr,
                attach_attr=self.attach_attr,
            )
        return H

    def _selected_nodes(self, G: nx.DiGraph) -> set[Hashable]:
        if self.selector == "uid":
            return {
                node
                for node, data in G.nodes(data=True)
                if data[self.uid_attr] in self.selected_values
            }
        return {
            node
            for node, data in G.nodes(data=True)
            if data[self.label_attr] in self.selected_values
        }

    def _selected_components(
        self,
        G: nx.DiGraph,
        selected: set[Hashable],
    ) -> list[tuple[Hashable, ...]]:
        """Return maximal selected components in deterministic rooted preorder."""

        roots: list[Hashable] = []
        for node in selected:
            parent = next(iter(G.predecessors(node)), None)
            if parent not in selected:
                roots.append(node)
        roots = deterministic_node_order(
            G,
            roots,
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
        )

        components: list[tuple[Hashable, ...]] = []
        seen: set[Hashable] = set()
        for root in roots:
            if root in seen:
                continue
            order: list[Hashable] = []
            stack = [root]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                order.append(node)
                children = [child for child in G.successors(node) if child in selected]
                children = deterministic_node_order(
                    G,
                    children,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                )
                stack.extend(reversed(children))
            components.append(tuple(order))

        if seen != selected:
            missing = selected - seen
            raise ValidationError(f"failed to enumerate selected nodes {missing!r}.")
        return components

    def _component_recipe(
        self,
        G: nx.DiGraph,
        component: Sequence[Hashable],
    ) -> tuple[tuple[int, ...], tuple[Token, ...], tuple[int, ...]]:
        index = {node: i for i, node in enumerate(component)}
        parent: list[int] = []
        labels: list[Token] = []

        for node in component:
            raw_parent = next(iter(G.predecessors(node)), None)
            parent.append(index[raw_parent] if raw_parent in index else -1)
            labels.append(base_token(G.nodes[node][self.label_attr]))

        parent_tuple = tuple(parent)
        if parent_tuple.count(-1) != 1:
            raise ValidationError(
                "a connected selected component must have exactly one exposed root."
            )
        if any(p >= i for i, p in enumerate(parent_tuple) if p >= 0):
            raise ValidationError("component canonical order must place parents before children.")

        # Every immediate recipe component is a one-site base token, so each
        # internal tree edge contributes the homogeneous attachment value 0.
        attach = tuple(0 for p in parent_tuple if p >= 0)
        return parent_tuple, tuple(labels), attach

    def _register_recipe(
        self,
        parent: tuple[int, ...],
        label: tuple[Token, ...],
        attach: tuple[int, ...],
    ) -> Token:
        token = named_component_token(self.selector, parent, label, attach)
        existing = self.vocab.entries.get(token)
        if existing is not None:
            if (
                existing.parent != parent
                or existing.label != label
                or existing.attach != attach
            ):
                raise ValidationError(
                    f"named-component token digest collision for token {token!r}."
                )
            return token

        step = len(self.vocab.creation_order)
        entry = VocabEntry(
            token=token,
            parent=parent,
            label=label,
            attach=attach,
            created_at_step=step,
            operation="component",
            metadata={
                "coarsener": "NamedVertexCoarsener",
                "selector": self.selector,
                "component_size": len(parent),
                "component_policy": self.component_policy,
            },
        )
        self.vocab.add(entry)

        dynamic_rule = EncodingRule(
            token=token,
            operation="component",
            created_at_step=step,
            pattern={
                "selector": self.selector,
                "component_size": len(parent),
            },
        )
        if isinstance(self.rules, list):
            self.rules.append(dynamic_rule)
        else:
            self.rules = tuple(self.rules) + (dynamic_rule,)
        return token


class NamedVertexCoarsener(TreeCoarsener):
    """Contract connected components induced by explicitly named UIDs or labels.

    Exactly one of ``uids`` and ``labels`` must be supplied. ``component_policy``
    determines whether every maximal matching component or only the largest one
    is contracted. Components of size one are left as base-token occurrences.

    ``fit`` performs no statistical learning. It only validates the supplied
    graphs (when validation is enabled) and constructs an encoder/decoder pair.
    Exact component recipes are added lazily to their shared vocabulary when
    ``transform`` first encounters them.
    """

    def __init__(
        self,
        *,
        uids: Collection[Hashable] | None = None,
        labels: Collection[str] | str | None = None,
        component_policy: ComponentPolicy = "all",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if (uids is None) == (labels is None):
            raise ValueError("supply exactly one of uids=... or labels=....")
        if component_policy not in {"all", "largest"}:
            raise ValueError("component_policy must be 'all' or 'largest'.")

        if uids is not None:
            values = tuple(uids)
            if not values:
                raise ValueError("uids must be nonempty.")
            if not all(isinstance(uid, HashableABC) for uid in values):
                raise TypeError("every selected UID must be hashable.")
            self.selector: SelectorKind = "uid"
            self.selected_values: frozenset[Hashable] = frozenset(values)
        else:
            label_values = (labels,) if isinstance(labels, str) else tuple(labels or ())
            if not label_values:
                raise ValueError("labels must be nonempty.")
            if not all(isinstance(label, str) for label in label_values):
                raise TypeError("every selected label must be a string.")
            self.selector = "label"
            self.selected_values = frozenset(label_values)

        self.component_policy = component_policy

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        if self.validate_inputs:
            for G in graphs:
                validate_raw_tree(
                    G,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    require_uid=False,
                )

        vocab = Vocabulary()
        dynamic_rules: list[EncodingRule] = []
        encoder = NamedVertexEncoder(
            model_id=self.model_id,
            vocab=vocab,
            rules=dynamic_rules,
            base_labels=frozenset(),
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            selector=self.selector,
            selected_values=self.selected_values,
            component_policy=self.component_policy,
        )
        decoder = StagedTreeDecoder(
            model_id=self.model_id,
            vocab=vocab,
            base_labels=frozenset(),
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
        )
        return encoder, decoder
