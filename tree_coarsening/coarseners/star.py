"""Star coarsener: contract repeated labeled sibling groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

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
from ..schema import RAW_INPUT_FLAG, encoded_node_attrs, max_component_time, normalize_coarsenable_tree
from ..stage_decoder import StructuralStageDecoder
from ..structural import CompositeType, infer_input_alphabet, structural_root_count
from ..validation import deterministic_node_order, validate_coarsenable_tree, validate_encoded_tree
from ..vocabulary import Token, VocabEntry, Vocabulary, normalize_attach_map


def star_token(parent_label: Token, child_label: Token, arity: int) -> tuple[Any, ...]:
    """Stable generic fitting label for a contracted sibling group."""

    if arity < 1:
        raise ValidationError("star arity must be positive.")
    return ("star", parent_label, child_label, int(arity))


@dataclass(frozen=True)
class StarRule:
    """Learned label pair and transform-time arities."""

    parent_label: Token
    child_label: Token
    support: int
    arities: tuple[int, ...]


@dataclass
class StarEncoder(TreeEncoder):
    """Apply learned sibling-label contractions to raw or encoded trees."""

    d: int = 2
    contract_d: int = 2
    star_rules: tuple[StarRule, ...] = ()

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

        target_child_labels: dict[Token, set[Token]] = defaultdict(set)
        for rule in self.star_rules:
            target_child_labels[rule.parent_label].add(rule.child_label)

        group_for_child: dict[Hashable, tuple[Hashable, int]] = {}
        group_info: dict[Hashable, tuple[Token, list[Hashable]]] = {}
        serial = 0

        for parent, data in G.nodes(data=True):
            parent_label = data[self.label_attr]
            allowed = target_child_labels.get(parent_label)
            if not allowed:
                continue
            by_label: dict[Token, list[Hashable]] = defaultdict(list)
            for child in G.successors(parent):
                child_label = G.nodes[child][self.label_attr]
                if child_label in allowed:
                    by_label[child_label].append(child)

            for child_label in sorted(by_label, key=repr):
                members = by_label[child_label]
                if len(members) < self.contract_d:
                    continue
                token = star_token(parent_label, child_label, len(members))
                if token not in self.vocab.entries:
                    # Keep a closed fitted vocabulary: unseen transform-time
                    # arities remain uncontracted.
                    continue
                ordered = deterministic_node_order(
                    G,
                    members,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                )
                group_node = ("__tc_star__", self.model_id, serial)
                serial += 1
                group_info[group_node] = (token, ordered)
                for position, child in enumerate(ordered):
                    if child in group_for_child:
                        raise ValidationError(
                            f"node {child!r} was assigned to more than one star group."
                        )
                    group_for_child[child] = (group_node, position)

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

        # Establish component offsets inside every contracted sibling token.
        for group_node, (_token, members) in group_info.items():
            site_cursor = 0
            root_cursor = 0
            for member in members:
                owner[member] = group_node
                site_offset[member] = site_cursor
                root_offset[member] = root_cursor
                site_cursor += G.nodes[member][self.size_attr]
                root_cursor += structural_root_count(G.nodes[member][self.type_attr], self.vocab)

        # Uncontracted occurrences preserve both their fitting label and exact type.
        for node, data in G.nodes(data=True):
            if node in group_for_child:
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

        # Contracted occurrences share a generic label while retaining exact
        # component labels/types in their structural type.
        for group_node, (token, members) in group_info.items():
            component_labels = tuple(G.nodes[m][self.label_attr] for m in members)
            component_types = tuple(G.nodes[m][self.type_attr] for m in members)
            component_sizes = tuple(G.nodes[m][self.size_attr] for m in members)
            component_roots = tuple(
                structural_root_count(G.nodes[m][self.type_attr], self.vocab)
                for m in members
            )
            exact_type = CompositeType(
                model_id=self.model_id,
                kind="star",
                label=token,
                parent=tuple(-1 for _ in members),
                component_labels=component_labels,
                component_types=component_types,
                component_sizes=component_sizes,
                component_root_counts=component_roots,
                attach=(),
            )
            uids = tuple(
                uid
                for member in members
                for uid in G.nodes[member][self.super_uid_attr]
            )
            H.add_node(
                group_node,
                **encoded_node_attrs(
                    label=token,
                    type_token=exact_type,
                    size=sum(component_sizes),
                    time=max_component_time(
                        *(G.nodes[member][self.time_attr] for member in members)
                    ),
                    super_label=tuple(
                        G.nodes[member][self.super_label_attr] for member in members
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

        # Project every input edge through owner/site/root coordinate maps.
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


class StarCoarsener(TreeCoarsener):
    """Learn frequent parent/child labels and contract matching sibling groups."""

    def __init__(
        self,
        d: int,
        m: int,
        *,
        contract_d: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if d < 2:
            raise ValueError("d should be at least 2 for a nontrivial star witness.")
        if m < 1:
            raise ValueError("m must be at least 1.")
        if contract_d is None:
            contract_d = d
        if contract_d < 2:
            raise ValueError("contract_d should be at least 2 for a nontrivial contraction.")
        if contract_d > d:
            raise ValueError("contract_d must be less than or equal to d.")
        self.d = d
        self.m = m
        self.contract_d = contract_d

    def _fit(self, graphs: list[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        input_alphabet = infer_input_alphabet(
            graphs,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            attach_attr=self.attach_attr,
        )
        pair_support: Counter[tuple[Token, Token]] = Counter()
        contract_arities: dict[tuple[Token, Token], Counter[int]] = defaultdict(Counter)

        for graph in graphs:
            for node, data in graph.nodes(data=True):
                parent_label = data[self.label_attr]
                counts = Counter(
                    graph.nodes[child][self.label_attr]
                    for child in graph.successors(node)
                )
                for child_label, count in counts.items():
                    if count >= self.d:
                        pair_support[(parent_label, child_label)] += 1
                    if count >= self.contract_d:
                        contract_arities[(parent_label, child_label)][count] += 1

        vocab = Vocabulary(symbols=input_alphabet)
        learned_rules: list[StarRule] = []
        encoding_rules: list[EncodingRule] = []
        step = 0
        for (parent_label, child_label), support in sorted(
            pair_support.items(), key=lambda item: (repr(item[0][0]), repr(item[0][1]))
        ):
            if support < self.m:
                continue
            arities = tuple(sorted(contract_arities[(parent_label, child_label)]))
            learned_rules.append(
                StarRule(
                    parent_label=parent_label,
                    child_label=child_label,
                    support=support,
                    arities=arities,
                )
            )
            for arity in arities:
                token = star_token(parent_label, child_label, arity)
                entry = VocabEntry(
                    token=token,
                    parent=tuple(-1 for _ in range(arity)),
                    label=tuple(child_label for _ in range(arity)),
                    attach=(),
                    created_at_step=step,
                    operation="siblings",
                    score=float(support),
                    metadata={
                        "coarsener": "StarCoarsener",
                        "parent_label": parent_label,
                        "child_label": child_label,
                        "arity": arity,
                        "support": support,
                        "d": self.d,
                        "m": self.m,
                        "contract_d": self.contract_d,
                    },
                )
                vocab.add(entry)
                encoding_rules.append(
                    EncodingRule(
                        token=token,
                        operation="siblings",
                        created_at_step=step,
                        pattern={
                            "parent_label": parent_label,
                            "child_label": child_label,
                            "arity": arity,
                            "witness_min_children": self.d,
                            "contract_min_children": self.contract_d,
                        },
                        score=float(support),
                    )
                )
                step += 1

        output_raw = all(graph.graph.get(RAW_INPUT_FLAG, False) for graph in graphs)
        encoder = StarEncoder(
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
            d=self.d,
            contract_d=self.contract_d,
            star_rules=tuple(learned_rules),
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
