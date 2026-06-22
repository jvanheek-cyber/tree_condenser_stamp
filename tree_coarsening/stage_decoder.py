"""Stage-local decoder for exact :class:`CompositeType` occurrences.

Unlike the legacy staged decoder, this decoder expands only structural types
owned by one fitted coarsener.  Earlier-stage types are opaque terminals, so a
BPE decoder can recover the preceding Star-encoded graph without understanding
or importing the Star vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal

import networkx as nx

from .decoder import DecodeBy, TreeDecoder
from .exceptions import ValidationError
from .nx_io import edge_attach_attrs, relabel_to_consecutive_topological
from .provenance import copy_graph_provenance, get_node_attrs_by_uid, normalize_super_uids
from .schema import encoded_node_attrs, max_component_time
from .structural import CompositeType, component_super_labels
from .validation import validate_encoded_tree, validate_raw_tree
from .vocabulary import Token, is_base_token, normalize_attach_map, raw_label_from_base_token


class _BoundaryExpansionRequired(Exception):
    def __init__(self, child: Hashable) -> None:
        self.child = child
        super().__init__(f"boundary child {child!r} must be expanded first")


@dataclass
class StructuralStageDecoder(TreeDecoder):
    """Decode exact types created by one coarsener stage.

    Parameters
    ----------
    output_raw:
        If true, a complete decode materializes raw UID-keyed vertices after all
        owned structural types have been expanded.  If false, complete decode
        returns the previous encoded-tree stage.
    """

    output_raw: bool = True

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
        if by not in {"node", "label", "type"}:
            raise ValueError("by must be 'node', 'label', or 'type'.")
        if boundary_policy not in {"expand", "raise"}:
            raise ValueError("boundary_policy must be 'expand' or 'raise'.")

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

        graph = H.copy(as_view=False)
        selected = self._select_targets(graph, target=target, by=by)
        if target is None:
            recursive = True

        self._decode_selected(
            graph,
            selected,
            recursive=recursive,
            boundary_policy=boundary_policy,
        )
        graph = relabel_to_consecutive_topological(graph)

        if target is None and self.output_raw:
            return self._materialize_raw(graph, validate=validate)

        if validate:
            validate_encoded_tree(
                graph,
                vocab=self.vocab,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
                attach_attr=self.attach_attr,
            )
        return graph

    def flatten_super_uids(self, token: Token, super_uids: Any) -> tuple[Any, ...]:
        return normalize_super_uids(super_uids)

    def _is_owned_type(self, type_token: Any) -> bool:
        return isinstance(type_token, CompositeType) and type_token.model_id == self.model_id

    def _select_targets(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | Token | None,
        by: DecodeBy,
    ) -> list[Hashable]:
        if target is None:
            return [
                node
                for node in nx.topological_sort(graph)
                if self._is_owned_type(graph.nodes[node][self.type_attr])
            ]
        if by == "node":
            if target not in graph:
                raise ValidationError(f"target node {target!r} is not in the encoded graph.")
            return [target]
        attr = self.label_attr if by == "label" else self.type_attr
        selected = [node for node, data in graph.nodes(data=True) if data[attr] == target]
        if not selected:
            raise ValidationError(f"no encoded nodes match {by}={target!r}.")
        return selected

    def _decode_selected(
        self,
        graph: nx.DiGraph,
        selected: list[Hashable],
        *,
        recursive: bool,
        boundary_policy: Literal["expand", "raise"],
    ) -> None:
        stack = list(selected)
        queued = set(stack)
        serial = 0
        while stack:
            node = stack.pop()
            queued.discard(node)
            if node not in graph:
                continue
            type_token = graph.nodes[node][self.type_attr]
            if not self._is_owned_type(type_token):
                continue
            try:
                new_nodes = self._expand_one(graph, node, serial=serial)
                serial += 1
            except _BoundaryExpansionRequired as requirement:
                child = requirement.child
                if boundary_policy == "raise":
                    raise ValidationError(
                        f"decoding node {node!r} would attach one collapsed child to "
                        "multiple exposed parent components."
                    ) from None
                if child not in graph or not self._is_owned_type(
                    graph.nodes[child][self.type_attr]
                ):
                    raise ValidationError(
                        f"boundary child {child!r} belongs to an earlier stage and cannot "
                        "be expanded by this decoder. This indicates inconsistent stage "
                        "attachment metadata."
                    ) from None
                # Expand the child first, then retry the current node.
                stack.append(node)
                stack.append(child)
                continue

            if recursive:
                for new_node in reversed(new_nodes):
                    if (
                        new_node in graph
                        and self._is_owned_type(graph.nodes[new_node][self.type_attr])
                        and new_node not in queued
                    ):
                        stack.append(new_node)
                        queued.add(new_node)

    def _expand_one(
        self,
        graph: nx.DiGraph,
        node: Hashable,
        *,
        serial: int,
    ) -> tuple[Hashable, ...]:
        data = graph.nodes[node]
        exact = data[self.type_attr]
        if not isinstance(exact, CompositeType) or exact.model_id != self.model_id:
            return ()

        flat_uids = normalize_super_uids(data[self.super_uid_attr])
        if len(flat_uids) != exact.site_count:
            raise ValidationError(
                f"node {node!r} stores {len(flat_uids)} UIDs, but its exact type "
                f"contains {exact.site_count} sites."
            )
        super_labels = component_super_labels(
            data[self.super_label_attr],
            component_sizes=exact.component_sizes,
            flat_uids=flat_uids,
        )
        uid_pieces: list[tuple[Any, ...]] = []
        cursor = 0
        for size in exact.component_sizes:
            uid_pieces.append(tuple(flat_uids[cursor : cursor + size]))
            cursor += size

        offsets: list[int] = []
        total = 0
        for size in exact.component_sizes:
            offsets.append(total)
            total += size

        def locate_site(site: int) -> tuple[int, int]:
            if site < 0 or site >= total:
                raise ValidationError(
                    f"outgoing attachment site {site} is outside 0..{total - 1}."
                )
            for component_i in range(len(offsets) - 1, -1, -1):
                if site >= offsets[component_i]:
                    return component_i, site - offsets[component_i]
            raise AssertionError("unreachable")

        # Preflight outgoing boundaries before mutating the graph.  If a child
        # must be expanded first, the caller can do so and safely retry.
        outgoing_routes: list[tuple[Hashable, int, tuple[int, ...]]] = []
        for _old_parent, outside_child, edge_data in list(graph.out_edges(node, data=True)):
            attach = normalize_attach_map(edge_data[self.attach_attr])
            routed = [locate_site(site) for site in attach]
            component_ids = {component_i for component_i, _local in routed}
            if len(component_ids) != 1:
                raise _BoundaryExpansionRequired(outside_child)
            outgoing_routes.append(
                (
                    outside_child,
                    routed[0][0],
                    tuple(local for _component, local in routed),
                )
            )

        provenance = get_node_attrs_by_uid(graph)
        component_nodes: list[Hashable] = []
        for i in range(exact.n_components):
            key = ("__tc_stage_decode__", self.model_id, node, serial, i)
            while key in graph:
                serial += 1
                key = ("__tc_stage_decode__", self.model_id, node, serial, i)
            component_nodes.append(key)
            uids = uid_pieces[i]
            times = [
                provenance[uid][self.time_attr]
                for uid in uids
                if uid in provenance and self.time_attr in provenance[uid]
            ]
            if len(times) != len(uids):
                raise ValidationError(
                    f"cannot recover all component times while decoding node {node!r}."
                )
            graph.add_node(
                key,
                **encoded_node_attrs(
                    label=exact.component_labels[i],
                    type_token=exact.component_types[i],
                    size=exact.component_sizes[i],
                    time=max_component_time(*times),
                    super_label=super_labels[i],
                    super_uids=uids,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                ),
            )

        for i, parent_i in enumerate(exact.parent):
            if parent_i == -1:
                continue
            graph.add_edge(
                component_nodes[parent_i],
                component_nodes[i],
                **edge_attach_attrs(exact.attachment_slice(i), attach_attr=self.attach_attr),
            )

        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            raise ValidationError(f"encoded node {node!r} has multiple parents.")
        if predecessors:
            outside_parent = predecessors[0]
            incoming = normalize_attach_map(graph.edges[outside_parent, node][self.attach_attr])
            if len(incoming) != exact.root_count:
                raise ValidationError(
                    f"incoming edge to {node!r} has {len(incoming)} roots; "
                    f"expected {exact.root_count}."
                )
            cursor = 0
            for i in exact.root_positions:
                width = exact.component_root_counts[i]
                piece = incoming[cursor : cursor + width]
                cursor += width
                graph.add_edge(
                    outside_parent,
                    component_nodes[i],
                    **edge_attach_attrs(piece, attach_attr=self.attach_attr),
                )
        elif exact.root_count != 1:
            raise ValidationError(
                f"root node {node!r} expands to {exact.root_count} roots rather than one."
            )

        for outside_child, component_i, local_map in outgoing_routes:
            graph.add_edge(
                component_nodes[component_i],
                outside_child,
                **edge_attach_attrs(local_map, attach_attr=self.attach_attr),
            )

        graph.remove_node(node)
        return tuple(component_nodes)

    def _materialize_raw(self, graph: nx.DiGraph, *, validate: bool) -> nx.DiGraph:
        provenance = get_node_attrs_by_uid(graph)
        out = nx.DiGraph()
        copy_graph_provenance(graph, out)
        node_uid: dict[Hashable, Any] = {}

        for node, data in graph.nodes(data=True):
            type_token = data[self.type_attr]
            if not is_base_token(type_token):
                raise ValidationError(
                    f"complete decode for model {self.model_id!r} stopped at non-base "
                    f"terminal type {type_token!r}. This decoder should have output_raw=False."
                )
            uids = normalize_super_uids(data[self.super_uid_attr])
            if len(uids) != 1:
                raise ValidationError(
                    f"base terminal node {node!r} stores {len(uids)} UIDs rather than one."
                )
            uid = uids[0]
            attrs = dict(provenance.get(uid, {}))
            attrs.setdefault(self.uid_attr, uid)
            attrs.setdefault(self.label_attr, raw_label_from_base_token(type_token))
            attrs.setdefault(self.time_attr, data[self.time_attr])
            out.add_node(uid, **attrs)
            node_uid[node] = uid

        for parent, child, edge_data in graph.edges(data=True):
            attach = normalize_attach_map(edge_data[self.attach_attr])
            if attach != (0,):
                raise ValidationError(
                    f"raw materialization encountered non-atomic attachment map {attach!r}."
                )
            out.add_edge(node_uid[parent], node_uid[child])

        if validate:
            validate_raw_tree(
                out,
                label_attr=self.label_attr,
                time_attr=self.time_attr,
                uid_attr=self.uid_attr,
                require_uid=True,
            )
        return out
