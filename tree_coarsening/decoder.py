"""Generic staged decoder for tree coarsening vocabularies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Hashable, Literal

import networkx as nx

from .exceptions import ValidationError
from .nx_io import edge_attach_attrs, relabel_to_consecutive_topological
from .schema import encoded_node_attrs, max_component_time
from .provenance import (
    copy_graph_provenance,
    get_node_attrs_by_uid,
    normalize_super_uids,
    split_super_uids,
)
from .vocabulary import (
    Token,
    Vocabulary,
    base_token,
    is_base_token,
    normalize_attach_map,
    raw_label_from_base_token,
)

DecodeBy = Literal["node", "label", "type"]


@dataclass
class TreeDecoder(ABC):
    """Abstract decoder artifact produced by ``TreeCoarsener.fit``."""

    model_id: str
    vocab: Vocabulary
    base_labels: frozenset[Token] = frozenset()

    label_attr: str = "label"
    type_attr: str = "type"
    size_attr: str = "size"
    time_attr: str = "time"
    uid_attr: str = "uid"
    super_label_attr: str = "super_label"
    super_uid_attr: str = "super_uids"
    attach_attr: str = "attach_map"

    @abstractmethod
    def decode(
        self,
        H: nx.DiGraph,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph:
        """Decode all or part of one encoded tree."""

    @abstractmethod
    def flatten_super_uids(self, token: Token, super_uids: Any) -> tuple[Any, ...]:
        """Return the occurrence's original UIDs in canonical site order."""


@dataclass(frozen=True)
class _ExpandedOccurrence:
    sites: tuple[Hashable, ...]
    roots: tuple[Hashable, ...]
    nodes: Mapping[Hashable, dict[str, Any]]
    edges: tuple[tuple[Hashable, Hashable], ...]


@dataclass(frozen=True)
class _EncodedDecomposition:
    """Replacement of one encoded node by encoded component nodes."""

    nodes: Mapping[Hashable, dict[str, Any]]
    edges: tuple[tuple[Hashable, Hashable, Mapping[str, Any]], ...]
    sites: tuple[tuple[Hashable, int], ...]
    roots: tuple[tuple[Hashable, int], ...]


@dataclass
class StagedTreeDecoder(TreeDecoder):
    """Generic decoder for staged ``(P, L, A)`` vocabularies."""

    def decode(
        self,
        H: nx.DiGraph,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph:
        from .validation import validate_encoded_tree

        if boundary_policy not in {"expand", "raise"}:
            raise ValueError("boundary_policy must be 'expand' or 'raise'.")
        if by == "type":
            # Legacy staged vocabularies are selected by token label.  Keep the
            # historical ``by='type'`` spelling as an alias without claiming
            # that modern encoded graphs lack an exact ``type`` field.
            by = "label"
        if by not in {"node", "label"}:
            raise ValueError("by must be 'node' or 'label'.")

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

        if target is None:
            return self._decode_full(H, validate=validate)

        return self._decode_partial(
            H,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def flatten_super_uids(self, token: Token, super_uids: Any) -> tuple[Any, ...]:
        return normalize_super_uids(super_uids)

    def _decode_full(self, H: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        from .validation import validate_raw_tree

        provenance = get_node_attrs_by_uid(H)
        expanded: dict[Hashable, _ExpandedOccurrence] = {}
        out = nx.DiGraph()

        for node, data in H.nodes(data=True):
            token = data[self.label_attr]
            super_uids = data[self.super_uid_attr]
            occurrence = self._expand_token(token, super_uids, provenance)
            expanded[node] = occurrence
            for raw_node, attrs in occurrence.nodes.items():
                if raw_node in out:
                    raise ValidationError(f"duplicate decoded uid/node {raw_node!r}.")
                out.add_node(raw_node, **dict(attrs))
            out.add_edges_from(occurrence.edges)

        for parent, child, data in H.edges(data=True):
            M = normalize_attach_map(data[self.attach_attr])
            parent_occ = expanded[parent]
            child_occ = expanded[child]
            if len(M) != len(child_occ.roots):
                raise ValidationError(
                    f"edge {(parent, child)!r} attach_map has length {len(M)}, "
                    f"but child exposes {len(child_occ.roots)} roots."
                )
            for root_i, parent_site_i in enumerate(M):
                out.add_edge(parent_occ.sites[parent_site_i], child_occ.roots[root_i])

        if validate:
            validate_raw_tree(
                out,
                label_attr=self.label_attr,
                time_attr=self.time_attr,
                uid_attr=self.uid_attr,
                require_uid=True,
            )
        return out

    def _decode_partial(
        self,
        H: nx.DiGraph,
        *,
        target: Hashable | Token,
        by: Literal["node", "label"],
        recursive: bool,
        boundary_policy: Literal["expand", "raise"],
        validate: bool,
    ) -> nx.DiGraph:
        from .validation import validate_encoded_tree

        selected = self._select_partial_targets(H, target=target, by=by)
        modes: dict[Hashable, Literal["one", "full"]] = {}
        for node in selected:
            token = H.nodes[node][self.label_attr]
            if is_base_token(token):
                continue
            modes[node] = "full" if recursive else "one"

        if not modes:
            return H.copy(as_view=False)

        modes = self._close_partial_boundary(H, modes, boundary_policy=boundary_policy)
        decomps = {node: self._decompose_node(H, node, mode=mode) for node, mode in modes.items()}

        out = nx.DiGraph()
        copy_graph_provenance(H, out)
        for node, data in H.nodes(data=True):
            if node not in decomps:
                out.add_node(node, **dict(data))

        for decomp in decomps.values():
            for node, attrs in decomp.nodes.items():
                if node in out:
                    raise ValidationError(f"partial decode produced duplicate node {node!r}.")
                out.add_node(node, **dict(attrs))
            for u, v, attrs in decomp.edges:
                out.add_edge(u, v, **dict(attrs))

        for u, v, data in H.edges(data=True):
            for new_u, new_v, attrs in self._rewire_edge(H, u, v, data, decomps):
                if new_u == new_v:
                    raise ValidationError(f"partial decode would create a self-loop on {new_u!r}.")
                if out.has_edge(new_u, new_v):
                    old = normalize_attach_map(out.edges[new_u, new_v][self.attach_attr])
                    new = normalize_attach_map(attrs[self.attach_attr])
                    if old != new:
                        raise ValidationError(
                            f"partial decode produced conflicting duplicate edge {(new_u, new_v)!r}: "
                            f"{old!r} versus {new!r}."
                        )
                    continue
                out.add_edge(new_u, new_v, **attrs)

        out = relabel_to_consecutive_topological(out)
        if validate:
            validate_encoded_tree(
                out,
                vocab=self.vocab,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
                attach_attr=self.attach_attr,
            )
        return out

    def _select_partial_targets(
        self, H: nx.DiGraph, *, target: Hashable | Token, by: Literal["node", "label"]
    ) -> set[Hashable]:
        if by == "node":
            if target not in H:
                raise ValidationError(f"target node {target!r} is not present in the graph.")
            return {target}
        matches = {node for node, data in H.nodes(data=True) if data[self.label_attr] == target}
        if not matches:
            raise ValidationError(f"no encoded nodes have label {target!r}.")
        return matches

    def _close_partial_boundary(
        self,
        H: nx.DiGraph,
        modes: Mapping[Hashable, Literal["one", "full"]],
        *,
        boundary_policy: Literal["expand", "raise"],
    ) -> dict[Hashable, Literal["one", "full"]]:
        closed = dict(modes)
        while True:
            decomps = {node: self._decompose_node(H, node, mode=mode) for node, mode in closed.items()}
            changed = False
            for u, v, data in H.edges(data=True):
                conflict_child = self._edge_boundary_conflict_child(H, u, v, data, decomps)
                if conflict_child is None:
                    continue
                if boundary_policy == "raise":
                    raise ValidationError(
                        "partial decode boundary cannot be represented as a directed tree: "
                        f"edge {(u, v)!r} would require multiple parents for one collapsed child."
                    )
                token = H.nodes[conflict_child][self.label_attr]
                if is_base_token(token):
                    raise ValidationError(
                        f"boundary conflict unexpectedly points to base token node {conflict_child!r}."
                    )
                if closed.get(conflict_child) != "full":
                    closed[conflict_child] = "full"
                    changed = True
            if not changed:
                return closed

    def _edge_boundary_conflict_child(
        self,
        H: nx.DiGraph,
        u: Hashable,
        v: Hashable,
        data: Mapping[str, Any],
        decomps: Mapping[Hashable, _EncodedDecomposition],
    ) -> Hashable | None:
        M = normalize_attach_map(data[self.attach_attr])
        parent_sites = self._site_map_for_node(H, u, decomps)
        child_roots = self._root_map_for_node(H, v, decomps)

        parents_by_child_component: dict[Hashable, set[Hashable]] = {}
        for root_i, parent_site_i in enumerate(M):
            parent_component, _ = parent_sites[parent_site_i]
            child_component, _ = child_roots[root_i]
            parents_by_child_component.setdefault(child_component, set()).add(parent_component)

        if any(len(parents) > 1 for parents in parents_by_child_component.values()):
            return v
        return None

    def _rewire_edge(
        self,
        H: nx.DiGraph,
        u: Hashable,
        v: Hashable,
        data: Mapping[str, Any],
        decomps: Mapping[Hashable, _EncodedDecomposition],
    ) -> tuple[tuple[Hashable, Hashable, dict[str, Any]], ...]:
        M = normalize_attach_map(data[self.attach_attr])
        parent_sites = self._site_map_for_node(H, u, decomps)
        child_roots = self._root_map_for_node(H, v, decomps)

        grouped: dict[tuple[Hashable, Hashable], list[int | None]] = {}
        for root_i, parent_site_i in enumerate(M):
            parent_component, local_site_i = parent_sites[parent_site_i]
            child_component, local_root_i = child_roots[root_i]
            child_token = self._component_token(H, child_component, decomps)
            child_root_count = self.vocab.root_count(child_token)
            slots = grouped.setdefault((parent_component, child_component), [None] * child_root_count)
            if slots[local_root_i] is not None and slots[local_root_i] != local_site_i:
                raise ValidationError(f"conflicting partial-decode attachment for edge {(u, v)!r}.")
            slots[local_root_i] = local_site_i

        out: list[tuple[Hashable, Hashable, dict[str, Any]]] = []
        for (parent_component, child_component), slots in grouped.items():
            if any(x is None for x in slots):
                raise ValidationError(
                    "partial decode produced an incomplete attach_map for edge "
                    f"{(parent_component, child_component)!r}: {slots!r}."
                )
            attach_map = tuple(int(x) for x in slots if x is not None)
            out.append(
                (parent_component, child_component, edge_attach_attrs(attach_map, attach_attr=self.attach_attr))
            )
        return tuple(out)

    def _site_map_for_node(
        self,
        H: nx.DiGraph,
        node: Hashable,
        decomps: Mapping[Hashable, _EncodedDecomposition],
    ) -> tuple[tuple[Hashable, int], ...]:
        if node in decomps:
            return decomps[node].sites
        token = H.nodes[node][self.label_attr]
        return tuple((node, i) for i in range(self.vocab.site_count(token)))

    def _root_map_for_node(
        self,
        H: nx.DiGraph,
        node: Hashable,
        decomps: Mapping[Hashable, _EncodedDecomposition],
    ) -> tuple[tuple[Hashable, int], ...]:
        if node in decomps:
            return decomps[node].roots
        token = H.nodes[node][self.label_attr]
        return tuple((node, i) for i in range(self.vocab.root_count(token)))

    def _component_token(
        self,
        H: nx.DiGraph,
        component: Hashable,
        decomps: Mapping[Hashable, _EncodedDecomposition],
    ) -> Token:
        if component in H and component not in decomps:
            return H.nodes[component][self.label_attr]
        for decomp in decomps.values():
            if component in decomp.nodes:
                return decomp.nodes[component][self.label_attr]
        raise ValidationError(f"unknown partial-decode component node {component!r}.")

    def _decompose_node(
        self,
        H: nx.DiGraph,
        node: Hashable,
        *,
        mode: Literal["one", "full"],
    ) -> _EncodedDecomposition:
        data = H.nodes[node]
        token = data[self.label_attr]
        if is_base_token(token):
            raise ValidationError(f"base token node {node!r} cannot be decomposed.")
        super_uids = data[self.super_uid_attr]
        provenance = get_node_attrs_by_uid(H)
        if mode == "full":
            return self._decompose_node_full(node, token, super_uids, provenance)
        if mode == "one":
            return self._decompose_node_one_level(node, token, super_uids, provenance)
        raise ValidationError(f"unknown partial decomposition mode {mode!r}.")

    def _decompose_node_one_level(
        self,
        source_node: Hashable,
        token: Token,
        super_uids: Any,
        provenance: Mapping[Any, Mapping[str, Any]],
    ) -> _EncodedDecomposition:
        entry = self.vocab.entries[token]
        pieces = split_super_uids(token, super_uids, self.vocab)

        nodes: dict[Hashable, dict[str, Any]] = {}
        edges: list[tuple[Hashable, Hashable, Mapping[str, Any]]] = []
        component_keys: list[Hashable] = []

        for i, child_token in enumerate(entry.label):
            key = self._partial_node_key(source_node, ("one", i))
            component_keys.append(key)
            piece = pieces[i]
            piece_times = [
                provenance[uid][self.time_attr]
                for uid in piece
                if uid in provenance and self.time_attr in provenance[uid]
            ]
            if len(piece_times) != len(piece):
                raise ValidationError(
                    f"cannot recover times for every UID while partially decoding {token!r}."
                )
            nodes[key] = encoded_node_attrs(
                label=child_token,
                type_token=child_token,
                size=len(piece),
                time=max_component_time(*piece_times),
                super_label=piece[0] if len(piece) == 1 else piece,
                super_uids=piece,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
            )

        attachment_slices = entry.attachment_slices(self.vocab)
        for i, p in enumerate(entry.parent):
            if p == -1:
                continue
            edges.append(
                (
                    component_keys[p],
                    component_keys[i],
                    edge_attach_attrs(attachment_slices[i], attach_attr=self.attach_attr),
                )
            )

        sites: list[tuple[Hashable, int]] = []
        for key, child_token in zip(component_keys, entry.label):
            for local_site_i in range(self.vocab.site_count(child_token)):
                sites.append((key, local_site_i))

        roots: list[tuple[Hashable, int]] = []
        for i, key in enumerate(component_keys):
            if entry.parent[i] == -1:
                child_token = entry.label[i]
                for local_root_i in range(self.vocab.root_count(child_token)):
                    roots.append((key, local_root_i))

        return _EncodedDecomposition(nodes=nodes, edges=tuple(edges), sites=tuple(sites), roots=tuple(roots))

    def _decompose_node_full(
        self,
        source_node: Hashable,
        token: Token,
        super_uids: Any,
        provenance: Mapping[Any, Mapping[str, Any]],
    ) -> _EncodedDecomposition:
        occurrence = self._expand_token(token, super_uids, provenance)
        nodes: dict[Hashable, dict[str, Any]] = {}
        key_by_uid: dict[Hashable, Hashable] = {}

        for uid in occurrence.sites:
            attrs = dict(occurrence.nodes[uid])
            raw_label = attrs.get(self.label_attr)
            if raw_label is None:
                raise ValidationError(f"missing raw label for expanded uid {uid!r}.")
            key = self._partial_node_key(source_node, ("full", uid))
            key_by_uid[uid] = key
            base_type = base_token(raw_label)
            nodes[key] = encoded_node_attrs(
                label=base_type,
                type_token=base_type,
                size=1,
                time=attrs.get(self.time_attr, 0.0),
                super_label=uid,
                super_uids=(uid,),
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
            )

        edges: list[tuple[Hashable, Hashable, Mapping[str, Any]]] = []
        for u, v in occurrence.edges:
            edges.append((key_by_uid[u], key_by_uid[v], edge_attach_attrs((0,), attach_attr=self.attach_attr)))

        sites = tuple((key_by_uid[uid], 0) for uid in occurrence.sites)
        roots = tuple((key_by_uid[uid], 0) for uid in occurrence.roots)
        return _EncodedDecomposition(nodes=nodes, edges=tuple(edges), sites=sites, roots=roots)

    def _partial_node_key(self, source_node: Hashable, path: tuple[Any, ...]) -> Hashable:
        return ("__tc_decoded__", self.model_id, source_node, path)

    def _expand_token(
        self,
        token: Token,
        super_uids: Any,
        provenance: Mapping[Any, Mapping[str, Any]],
    ) -> _ExpandedOccurrence:
        if is_base_token(token):
            uids = normalize_super_uids(super_uids)
            if len(uids) != 1:
                raise ValidationError(f"base token {token!r} expects one UID, got {len(uids)}.")
            uid = uids[0]
            attrs = dict(provenance.get(uid, {}))
            attrs.setdefault(self.label_attr, raw_label_from_base_token(token))
            attrs.setdefault(self.uid_attr, uid)
            if self.time_attr not in attrs:
                attrs[self.time_attr] = 0.0
            return _ExpandedOccurrence(sites=(uid,), roots=(uid,), nodes={uid: attrs}, edges=())

        if token not in self.vocab.entries:
            raise ValidationError(f"unknown learned token {token!r}.")
        entry = self.vocab.entries[token]
        pieces = split_super_uids(token, super_uids, self.vocab)

        components: list[_ExpandedOccurrence] = []
        nodes: dict[Hashable, dict[str, Any]] = {}
        edges: list[tuple[Hashable, Hashable]] = []
        for i, child_token in enumerate(entry.label):
            occ = self._expand_token(child_token, pieces[i], provenance)
            components.append(occ)
            for node, attrs in occ.nodes.items():
                if node in nodes:
                    raise ValidationError(f"duplicate uid {node!r} inside token {token!r}.")
                nodes[node] = dict(attrs)
            edges.extend(occ.edges)

        attachment_slices = entry.attachment_slices(self.vocab)
        for i, p in enumerate(entry.parent):
            if p == -1:
                continue
            attach = attachment_slices[i]
            parent_occ = components[p]
            child_occ = components[i]
            if len(attach) != len(child_occ.roots):
                raise ValidationError(
                    f"entry {token!r} attachment slice for position {i} has length mismatch."
                )
            for root_i, parent_site_i in enumerate(attach):
                edges.append((parent_occ.sites[parent_site_i], child_occ.roots[root_i]))

        sites = tuple(site for occ in components for site in occ.sites)
        roots = tuple(
            root
            for i, occ in enumerate(components)
            if entry.parent[i] == -1
            for root in occ.roots
        )
        return _ExpandedOccurrence(sites=sites, roots=roots, nodes=nodes, edges=tuple(edges))


@dataclass
class LazyTreeDecoder(TreeDecoder):
    """Lazy composition of fitted decoders in reverse application order."""

    decoders: tuple[TreeDecoder, ...] = ()

    def decode(
        self,
        H: nx.DiGraph,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph:
        if target is not None:
            raise NotImplementedError(
                "targeted partial decode for a lazy combined decoder is stage-local; "
                "call the relevant component decoder directly."
            )
        out = H
        for decoder in reversed(self.decoders):
            out = decoder.decode(
                out,
                target=None,
                by=by,
                recursive=recursive,
                boundary_policy=boundary_policy,
                validate=validate,
            )
        return out

    def flatten_super_uids(self, token: Token, super_uids: Any) -> tuple[Any, ...]:
        return normalize_super_uids(super_uids)
