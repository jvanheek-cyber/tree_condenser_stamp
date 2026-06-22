"""Deterministic contraction of connected components selected by UID or label."""

from __future__ import annotations

from collections.abc import Collection, Hashable as HashableABC, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Hashable, Literal

import networkx as nx

from ..coarsener import TreeCoarsener
from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..nx_io import edge_attach_attrs, relabel_to_consecutive_topological
from ..provenance import (
    PROVENANCE_KEY,
    copy_graph_provenance,
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
from ..validation import (
    deterministic_node_order,
    validate_coarsenable_tree,
    validate_encoded_tree,
)
from ..vocabulary import Token, TokenSpec, VocabEntry, Vocabulary, normalize_attach_map

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

        selected = self._selected_nodes(G)
        components = self._selected_components(G, selected)
        components = [component for component in components if len(component) >= 2]
        if self.component_policy == "largest" and components:
            components = [max(components, key=len)]

        component_for_node: dict[Hashable, tuple[Hashable, int]] = {}
        component_info: dict[
            Hashable,
            tuple[
                Token,
                tuple[Hashable, ...],
                tuple[int, ...],
                tuple[int, ...],
            ],
        ] = {}

        for serial, component in enumerate(components):
            for node in component:
                self.vocab.add_symbol(
                    G.nodes[node][self.label_attr],
                    TokenSpec(
                        site_count=G.nodes[node][self.size_attr],
                        root_count=structural_root_count(
                            G.nodes[node][self.type_attr], self.vocab
                        ),
                    ),
                )
            parent, labels, attach = self._component_recipe(G, component)
            token = self._register_recipe(parent, labels, attach)
            coarse_node = ("__tc_named_component__", self.model_id, serial)
            component_info[coarse_node] = (token, component, parent, attach)
            for position, node in enumerate(component):
                if node in component_for_node:
                    raise ValidationError(
                        f"node {node!r} belongs to more than one selected component."
                    )
                component_for_node[node] = (coarse_node, position)

        H = nx.DiGraph()
        if get_node_attrs_by_uid(G):
            copy_graph_provenance(G, H)
        else:
            H.graph[PROVENANCE_KEY] = provenance_from_raw_graph(G, uid_attr=self.uid_attr)
        H.graph[RAW_INPUT_FLAG] = False
        H.graph["tree_coarsening_schema"] = {
            "schema_version": "0.3",
            "model_id": self.model_id,
            "node_label_semantics": "fit symbol",
            "node_type_semantics": "exact structural variant",
        }

        owner: dict[Hashable, Hashable] = {}
        site_offset: dict[Hashable, int] = {}
        root_offset: dict[Hashable, int] = {}

        for coarse_node, (_token, component, parent_recipe, _attach) in component_info.items():
            site_cursor = 0
            root_cursor = 0
            for position, node in enumerate(component):
                owner[node] = coarse_node
                site_offset[node] = site_cursor
                site_cursor += G.nodes[node][self.size_attr]
                if parent_recipe[position] == -1:
                    root_offset[node] = root_cursor
                    root_cursor += structural_root_count(
                        G.nodes[node][self.type_attr], self.vocab
                    )
                else:
                    root_offset[node] = 0

        for node, data in G.nodes(data=True):
            if node in component_for_node:
                continue
            owner[node] = node
            site_offset[node] = 0
            root_offset[node] = 0
            H.add_node(
                node,
                **encoded_node_attrs(
                    label=data[self.label_attr],
                    type_token=data[self.type_attr],
                    size=data[self.size_attr],
                    time=data[self.time_attr],
                    super_label=data[self.super_label_attr],
                    super_uids=data[self.super_uid_attr],
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                ),
            )

        for coarse_node, (token, component, parent_recipe, attach) in component_info.items():
            component_labels = tuple(G.nodes[node][self.label_attr] for node in component)
            component_types = tuple(G.nodes[node][self.type_attr] for node in component)
            component_sizes = tuple(G.nodes[node][self.size_attr] for node in component)
            component_roots = tuple(
                structural_root_count(G.nodes[node][self.type_attr], self.vocab)
                for node in component
            )
            exact_type = CompositeType(
                model_id=self.model_id,
                kind="component",
                label=token,
                parent=parent_recipe,
                component_labels=component_labels,
                component_types=component_types,
                component_sizes=component_sizes,
                component_root_counts=component_roots,
                attach=attach,
            )
            uids = tuple(
                uid
                for node in component
                for uid in G.nodes[node][self.super_uid_attr]
            )
            H.add_node(
                coarse_node,
                **encoded_node_attrs(
                    label=token,
                    type_token=exact_type,
                    size=sum(component_sizes),
                    time=max_component_time(
                        *(G.nodes[node][self.time_attr] for node in component)
                    ),
                    super_label=tuple(
                        G.nodes[node][self.super_label_attr] for node in component
                    ),
                    super_uids=uids,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                ),
            )

        edge_maps: dict[tuple[Hashable, Hashable], list[int | None]] = {}
        for parent, child, edge_data in G.edges(data=True):
            coarse_parent = owner[parent]
            coarse_child = owner[child]
            if coarse_parent == coarse_child:
                continue
            incoming = normalize_attach_map(edge_data[self.attach_attr])
            child_roots = structural_root_count(G.nodes[child][self.type_attr], self.vocab)
            if len(incoming) != child_roots:
                raise ValidationError(
                    f"edge {(parent, child)!r} carries {len(incoming)} roots; "
                    f"child type expects {child_roots}."
                )
            coarse_child_roots = structural_root_count(
                H.nodes[coarse_child][self.type_attr], self.vocab
            )
            slots = edge_maps.setdefault(
                (coarse_parent, coarse_child), [None] * coarse_child_roots
            )
            root_start = root_offset[child]
            parent_start = site_offset[parent]
            for local_root, parent_site in enumerate(incoming):
                root_index = root_start + local_root
                translated_site = parent_start + parent_site
                old = slots[root_index]
                if old is not None and old != translated_site:
                    raise ValidationError(
                        f"conflicting projected attachment for coarse edge "
                        f"{(coarse_parent, coarse_child)!r}, root {root_index}."
                    )
                slots[root_index] = translated_site

        for (parent, child), slots in edge_maps.items():
            if any(site is None for site in slots):
                raise ValidationError(
                    f"incomplete attach_map for coarse edge {(parent, child)!r}: {slots!r}."
                )
            H.add_edge(
                parent,
                child,
                **edge_attach_attrs(
                    tuple(int(site) for site in slots if site is not None),
                    attach_attr=self.attach_attr,
                ),
            )

        H = relabel_to_consecutive_topological(H)
        if validate:
            validate_encoded_tree(
                H,
                vocab=self.vocab,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
                attach_attr=self.attach_attr,
            )
        return H

    def _selected_nodes(self, G: nx.DiGraph) -> set[Hashable]:
        if self.selector == "uid":
            return {
                node
                for node, data in G.nodes(data=True)
                if any(uid in self.selected_values for uid in data[self.super_uid_attr])
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
            labels.append(G.nodes[node][self.label_attr])

        parent_tuple = tuple(parent)
        if parent_tuple.count(-1) != 1:
            raise ValidationError(
                "a connected selected component must have exactly one exposed root."
            )
        if any(p >= i for i, p in enumerate(parent_tuple) if p >= 0):
            raise ValidationError("component canonical order must place parents before children.")

        attach_values: list[int] = []
        for position, parent_i in enumerate(parent_tuple):
            if parent_i == -1:
                continue
            parent_node = component[parent_i]
            child_node = component[position]
            attach_values.extend(
                normalize_attach_map(G.edges[parent_node, child_node][self.attach_attr])
            )
        attach = tuple(attach_values)
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
        labels: Collection[Hashable] | str | None = None,
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
            if not all(isinstance(label, HashableABC) for label in label_values):
                raise TypeError("every selected label must be hashable.")
            self.selector = "label"
            self.selected_values = frozenset(label_values)

        self.component_policy = component_policy

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        if self.validate_inputs:
            for G in graphs:
                validate_coarsenable_tree(
                    G,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                )

        input_alphabet = infer_input_alphabet(
            graphs,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            attach_attr=self.attach_attr,
        )
        vocab = Vocabulary(symbols=input_alphabet)
        dynamic_rules: list[EncodingRule] = []
        encoder = NamedVertexEncoder(
            model_id=self.model_id,
            vocab=vocab,
            rules=dynamic_rules,
            base_labels=frozenset(),
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_label_attr=self.super_label_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            selector=self.selector,
            selected_values=self.selected_values,
            component_policy=self.component_policy,
        )
        output_raw = all(graph.graph.get(RAW_INPUT_FLAG, False) for graph in graphs)
        decoder = StructuralStageDecoder(
            model_id=self.model_id,
            vocab=vocab,
            base_labels=frozenset(),
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
