"""Star coarsener: contract repeated labeled child starbursts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

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


def star_token(parent_label: str, child_label: str, arity: int) -> tuple[str, str, str, int]:
    """Stable token id for a contracted star.

    ``('star', P, C, k)`` means that ``k`` children with raw label ``C`` were
    contracted under an encoded parent whose raw label is ``P``.
    """

    if arity < 1:
        raise ValidationError("star arity must be positive.")
    return ("star", parent_label, child_label, int(arity))


@dataclass(frozen=True)
class StarRule:
    """Learned pair ``(P, C)`` and the arities supported by the fixed vocabulary."""

    parent_label: str
    child_label: str
    support: int
    arities: tuple[int, ...]


@dataclass
class StarEncoder(TreeEncoder):
    """Encoder for ``StarCoarsener``.

    For every learned pair ``(P, C)``, a node labeled ``P`` with at least
    ``contract_d`` children labeled ``C`` has those matching children contracted
    when the resulting arity is present in the fitted vocabulary.
    """

    d: int = 2
    contract_d: int = 2
    star_rules: tuple[StarRule, ...] = ()

    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        if validate:
            validate_raw_tree(
                G,
                label_attr=self.label_attr,
                time_attr=self.time_attr,
                uid_attr=self.uid_attr,
                require_uid=False,
            )
        G = copy_with_uids(G, uid_attr=self.uid_attr)

        target_child_labels: dict[str, set[str]] = defaultdict(set)
        for rule in self.star_rules:
            target_child_labels[rule.parent_label].add(rule.child_label)

        group_for_child: dict[Hashable, tuple[Hashable, int]] = {}
        group_info: dict[Hashable, tuple[Token, list[Hashable]]] = {}

        serial = 0
        for parent, data in G.nodes(data=True):
            parent_label = data[self.label_attr]
            allowed_child_labels = target_child_labels.get(parent_label)
            if not allowed_child_labels:
                continue

            by_child_label: dict[str, list[Hashable]] = defaultdict(list)
            for child in G.successors(parent):
                child_label = G.nodes[child][self.label_attr]
                if child_label in allowed_child_labels:
                    by_child_label[child_label].append(child)

            for child_label in sorted(by_child_label):
                members = by_child_label[child_label]
                if len(members) < self.contract_d:
                    continue
                token = star_token(parent_label, child_label, len(members))
                if token not in self.vocab.entries:
                    # The learned vocabulary is fixed after fit. A transform-time
                    # arity not present during fit is left uncontracted.
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
                group_info[group_node] = (token, list(ordered))
                for i, child in enumerate(ordered):
                    if child in group_for_child:
                        raise ValidationError(f"node {child!r} assigned to more than one star group.")
                    group_for_child[child] = (group_node, i)

        H = nx.DiGraph()
        H.graph[PROVENANCE_KEY] = provenance_from_raw_graph(G, uid_attr=self.uid_attr)
        H.graph["tree_coarsening_schema"] = {
            "schema_version": "0.2",
            "model_id": self.model_id,
            "node_label_semantics": "encoded token id",
            "super_uid_attr": self.super_uid_attr,
            "attach_attr": self.attach_attr,
        }

        owner: dict[Hashable, Hashable] = {}
        site_in_owner: dict[Hashable, int] = {}
        root_index_in_owner: dict[Hashable, int] = {}

        # Base occurrences not swallowed by a star token stay as base tokens.
        for node, data in G.nodes(data=True):
            if node in group_for_child:
                group_node, i = group_for_child[node]
                owner[node] = group_node
                site_in_owner[node] = i
                root_index_in_owner[node] = i
                continue

            token = base_token(data[self.label_attr])
            owner[node] = node
            site_in_owner[node] = 0
            root_index_in_owner[node] = 0
            H.add_node(
                node,
                **{
                    self.label_attr: token,
                    self.super_uid_attr: (data[self.uid_attr],),
                },
            )

        # Add contracted sibling-token occurrences.
        for group_node, (token, members) in group_info.items():
            uids = tuple(G.nodes[v][self.uid_attr] for v in members)
            H.add_node(group_node, **{self.label_attr: token, self.super_uid_attr: uids})

        # Rewire raw edges using owner/root/site coordinate maps.
        edge_maps: dict[tuple[Hashable, Hashable], list[int | None]] = {}
        for u, v in G.edges:
            ou = owner[u]
            ov = owner[v]
            if ou == ov:
                continue
            child_token = H.nodes[ov][self.label_attr]
            n_roots = self.vocab.root_count(child_token)
            key = (ou, ov)
            M = edge_maps.setdefault(key, [None] * n_roots)
            root_i = root_index_in_owner[v]
            site_i = site_in_owner[u]
            if M[root_i] is not None and M[root_i] != site_i:
                raise ValidationError(f"conflicting attachment for coarse edge {key!r}, root {root_i}.")
            M[root_i] = site_i

        for (u, v), maybe_map in edge_maps.items():
            if any(x is None for x in maybe_map):
                raise ValidationError(f"incomplete attach_map for coarse edge {(u, v)!r}: {maybe_map!r}")
            attach_map = tuple(int(x) for x in maybe_map if x is not None)
            H.add_edge(u, v, **edge_attach_attrs(attach_map, attach_attr=self.attach_attr))

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


class StarCoarsener(TreeCoarsener):
    """Simple repeated-star coarsener.

    A pair ``(P, C)`` is learned when at least ``m`` vertices labeled ``P`` have
    at least ``d`` children labeled ``C``. After the pair is learned, transform
    contracts all matching ``C`` children under any ``P`` vertex with at least
    ``contract_d`` such children, provided the resulting arity is present in the
    fitted vocabulary.
    """

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
        pair_support: Counter[tuple[str, str]] = Counter()
        contract_arities: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
        base_labels: set[str] = set()

        for G in graphs:
            if self.validate_inputs:
                validate_raw_tree(
                    G,
                    label_attr=self.label_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    require_uid=False,
                )
            for node, data in G.nodes(data=True):
                parent_label = data[self.label_attr]
                base_labels.add(parent_label)
                counts = Counter(G.nodes[child][self.label_attr] for child in G.successors(node))
                for child_label, count in counts.items():
                    if count >= self.d:
                        pair_support[(parent_label, child_label)] += 1
                    if count >= self.contract_d:
                        contract_arities[(parent_label, child_label)][count] += 1

        vocab = Vocabulary()
        learned_rules: list[StarRule] = []
        encoding_rules: list[EncodingRule] = []
        step = 0

        for (parent_label, child_label), support in sorted(pair_support.items()):
            if support < self.m:
                continue
            arities = tuple(sorted(contract_arities[(parent_label, child_label)]))
            rule = StarRule(
                parent_label=parent_label,
                child_label=child_label,
                support=support,
                arities=arities,
            )
            learned_rules.append(rule)

            for arity in arities:
                token = star_token(parent_label, child_label, arity)
                entry = VocabEntry(
                    token=token,
                    parent=tuple(-1 for _ in range(arity)),
                    label=tuple(base_token(child_label) for _ in range(arity)),
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

        encoder = StarEncoder(
            model_id=self.model_id,
            vocab=vocab,
            rules=tuple(encoding_rules),
            base_labels=frozenset(base_labels),
            label_attr=self.label_attr,
            time_attr=self.time_attr,
            uid_attr=self.uid_attr,
            super_uid_attr=self.super_uid_attr,
            attach_attr=self.attach_attr,
            d=self.d,
            contract_d=self.contract_d,
            star_rules=tuple(learned_rules),
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
