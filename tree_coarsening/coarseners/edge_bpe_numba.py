"""Experimental Numba training backend for edge-only tree BPE.

This module accelerates the mutable forest and incremental pair-index updates
used while fitting :class:`~tree_coarsening.EdgeBPECoarsener`.  NetworkX
validation/conversion, vocabulary construction, and the public encoder/decoder
remain ordinary Python.

The backend is optional.  Importing :mod:`tree_coarsening` does not require
Numba; requesting ``backend="numba"`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .edge_bpe import EdgeKey, _CompactEdgeTree, _TokenCodec

try:  # pragma: no cover - availability is environment-dependent
    from numba import njit, types
    from numba.typed import Dict, List
except ImportError as exc:  # pragma: no cover
    njit = None
    types = None
    Dict = None
    List = None
    _NUMBA_IMPORT_ERROR: ImportError | None = exc
else:
    _NUMBA_IMPORT_ERROR = None


def numba_available() -> bool:
    """Return whether the optional Numba backend can be constructed."""

    return _NUMBA_IMPORT_ERROR is None


def require_numba() -> None:
    """Raise an informative error when the optional dependency is absent."""

    if _NUMBA_IMPORT_ERROR is not None:
        raise ImportError(
            "The experimental Numba backend requires the optional dependency. "
            "Install tree-coarsening[numba] or install numba directly."
        ) from _NUMBA_IMPORT_ERROR


if _NUMBA_IMPORT_ERROR is None:
    _PAIR_KEY_TYPE = types.UniTuple(types.int64, 3)

    @njit(cache=True)
    def _add_edge(
        child: int,
        parent: np.ndarray,
        label: np.ndarray,
        attach: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
        key_attach: Any,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> None:
        p = parent[child]
        key = (label[p], label[child], attach[child])
        if key in pair_to_bucket:
            bucket = pair_to_bucket[key]
        else:
            bucket = len(bucket_count)
            pair_to_bucket[key] = bucket
            key_parent.append(key[0])
            key_child.append(key[1])
            key_attach.append(key[2])
            bucket_count.append(0)
            bucket_head.append(-1)

        old_head = bucket_head[bucket]
        edge_bucket[child] = bucket
        edge_prev[child] = -1
        edge_next[child] = old_head
        if old_head != -1:
            edge_prev[old_head] = child
        bucket_head[bucket] = child
        bucket_count[bucket] += 1

    @njit(cache=True)
    def _remove_edge(
        child: int,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> None:
        bucket = edge_bucket[child]
        if bucket < 0:
            raise RuntimeError("attempted to remove an unindexed edge")
        previous = edge_prev[child]
        following = edge_next[child]
        if previous == -1:
            bucket_head[bucket] = following
        else:
            edge_next[previous] = following
        if following != -1:
            edge_prev[following] = previous
        edge_bucket[child] = -1
        edge_prev[child] = -1
        edge_next[child] = -1
        bucket_count[bucket] -= 1
        if bucket_count[bucket] < 0:
            raise RuntimeError("pair count became negative")

    @njit(cache=True)
    def _build_pair_index(
        parent: np.ndarray,
        label: np.ndarray,
        attach: np.ndarray,
        alive: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        pair_to_bucket = Dict.empty(key_type=_PAIR_KEY_TYPE, value_type=types.int64)
        key_parent = List.empty_list(types.int64)
        key_child = List.empty_list(types.int64)
        key_attach = List.empty_list(types.int64)
        bucket_count = List.empty_list(types.int64)
        bucket_head = List.empty_list(types.int64)
        n = parent.size
        edge_bucket = np.full(n, -1, dtype=np.int64)
        edge_prev = np.full(n, -1, dtype=np.int64)
        edge_next = np.full(n, -1, dtype=np.int64)
        for child in range(n):
            p = parent[child]
            if alive[child] and p >= 0 and alive[p]:
                _add_edge(
                    child,
                    parent,
                    label,
                    attach,
                    pair_to_bucket,
                    key_parent,
                    key_child,
                    key_attach,
                    bucket_count,
                    bucket_head,
                    edge_bucket,
                    edge_prev,
                    edge_next,
                )
        return (
            pair_to_bucket,
            key_parent,
            key_child,
            key_attach,
            bucket_count,
            bucket_head,
            edge_bucket,
            edge_prev,
            edge_next,
        )

    @njit(cache=True)
    def _max_count_buckets(
        minimum: int,
        key_parent: Any,
        key_child: Any,
        key_attach: Any,
        bucket_count: Any,
    ) -> tuple[int, np.ndarray]:
        best_count = 0
        for bucket in range(len(bucket_count)):
            count = bucket_count[bucket]
            if count >= minimum and count > best_count:
                best_count = count
        if best_count == 0:
            return 0, np.empty((0, 4), dtype=np.int64)

        n_best = 0
        for bucket in range(len(bucket_count)):
            if bucket_count[bucket] == best_count:
                n_best += 1
        rows = np.empty((n_best, 4), dtype=np.int64)
        cursor = 0
        for bucket in range(len(bucket_count)):
            if bucket_count[bucket] == best_count:
                rows[cursor, 0] = bucket
                rows[cursor, 1] = key_parent[bucket]
                rows[cursor, 2] = key_child[bucket]
                rows[cursor, 3] = key_attach[bucket]
                cursor += 1
        return best_count, rows

    @njit(cache=True)
    def _bucket_occurrences(
        bucket: int,
        bucket_count: Any,
        bucket_head: Any,
        edge_next: np.ndarray,
    ) -> np.ndarray:
        expected = bucket_count[bucket]
        out = np.empty(expected, dtype=np.int64)
        cursor = 0
        child = bucket_head[bucket]
        while child != -1:
            if cursor >= expected:
                raise RuntimeError("pair bucket contains more edges than its count")
            out[cursor] = child
            cursor += 1
            child = edge_next[child]
        if cursor != expected:
            raise RuntimeError("pair bucket contains fewer edges than its count")
        return out

    @njit(cache=True)
    def _contract_one(
        parent_node: int,
        child_node: int,
        new_label: int,
        parent: np.ndarray,
        first_child: np.ndarray,
        last_child: np.ndarray,
        next_sibling: np.ndarray,
        prev_sibling: np.ndarray,
        label: np.ndarray,
        time: np.ndarray,
        attach: np.ndarray,
        alive: np.ndarray,
        site_count: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
        key_attach: Any,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> None:
        grandparent = parent[parent_node]
        parent_site_count = site_count[label[parent_node]]

        if grandparent != -1:
            _remove_edge(
                parent_node,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )

        current = first_child[parent_node]
        found_child = False
        while current != -1:
            following = next_sibling[current]
            _remove_edge(
                current,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )
            if current == child_node:
                found_child = True
            current = following
        if not found_child:
            raise RuntimeError("contracted child is missing from its parent's child list")

        current = first_child[child_node]
        while current != -1:
            following = next_sibling[current]
            _remove_edge(
                current,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )
            current = following

        # Remove child_node from parent_node's sibling-linked child list.
        previous = prev_sibling[child_node]
        following = next_sibling[child_node]
        if previous == -1:
            first_child[parent_node] = following
        else:
            next_sibling[previous] = following
        if following == -1:
            last_child[parent_node] = previous
        else:
            prev_sibling[following] = previous
        prev_sibling[child_node] = -1
        next_sibling[child_node] = -1

        child_first = first_child[child_node]
        child_last = last_child[child_node]
        current = child_first
        while current != -1:
            parent[current] = parent_node
            attach[current] += parent_site_count
            current = next_sibling[current]

        # Preserve the Python backend's order: surviving parent children first,
        # followed by the removed child's children.
        if child_first != -1:
            old_last = last_child[parent_node]
            if old_last == -1:
                first_child[parent_node] = child_first
                prev_sibling[child_first] = -1
            else:
                next_sibling[old_last] = child_first
                prev_sibling[child_first] = old_last
            last_child[parent_node] = child_last

        label[parent_node] = new_label
        if time[child_node] < time[parent_node]:
            time[parent_node] = time[child_node]

        alive[child_node] = False
        parent[child_node] = -1
        first_child[child_node] = -1
        last_child[child_node] = -1
        attach[child_node] = -1

        if grandparent != -1:
            _add_edge(
                parent_node,
                parent,
                label,
                attach,
                pair_to_bucket,
                key_parent,
                key_child,
                key_attach,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )
        current = first_child[parent_node]
        while current != -1:
            _add_edge(
                current,
                parent,
                label,
                attach,
                pair_to_bucket,
                key_parent,
                key_child,
                key_attach,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )
            current = next_sibling[current]

    @njit(cache=True)
    def _contract_candidates(
        candidates: np.ndarray,
        selected_parent_label: int,
        selected_child_label: int,
        selected_attach: int,
        new_label: int,
        epoch: int,
        used_epoch: np.ndarray,
        parent: np.ndarray,
        first_child: np.ndarray,
        last_child: np.ndarray,
        next_sibling: np.ndarray,
        prev_sibling: np.ndarray,
        label: np.ndarray,
        time: np.ndarray,
        attach: np.ndarray,
        alive: np.ndarray,
        site_count: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
        key_attach: Any,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> int:
        events = 0
        n = parent.size
        for cursor in range(candidates.size):
            child = candidates[cursor]
            if child < 0 or child >= n or not alive[child]:
                continue
            p = parent[child]
            if p < 0 or not alive[p]:
                continue
            if used_epoch[p] == epoch or used_epoch[child] == epoch:
                continue
            if (
                label[p] != selected_parent_label
                or label[child] != selected_child_label
                or attach[child] != selected_attach
            ):
                continue
            _contract_one(
                p,
                child,
                new_label,
                parent,
                first_child,
                last_child,
                next_sibling,
                prev_sibling,
                label,
                time,
                attach,
                alive,
                site_count,
                pair_to_bucket,
                key_parent,
                key_child,
                key_attach,
                bucket_count,
                bucket_head,
                edge_bucket,
                edge_prev,
                edge_next,
            )
            used_epoch[p] = epoch
            used_epoch[child] = epoch
            events += 1
        return events

    @njit(cache=True)
    def _full_recount(
        parent: np.ndarray,
        label: np.ndarray,
        attach: np.ndarray,
        alive: np.ndarray,
    ) -> Any:
        recount = Dict.empty(key_type=_PAIR_KEY_TYPE, value_type=types.int64)
        for child in range(parent.size):
            p = parent[child]
            if alive[child] and p >= 0 and alive[p]:
                key = (label[p], label[child], attach[child])
                recount[key] = recount.get(key, 0) + 1
        return recount


@dataclass(slots=True)
class NumbaTrainingForest:
    """One flattened forest plus an incremental compiled pair index."""

    parent: np.ndarray
    first_child: np.ndarray
    last_child: np.ndarray
    next_sibling: np.ndarray
    prev_sibling: np.ndarray
    label: np.ndarray
    time: np.ndarray
    attach: np.ndarray
    alive: np.ndarray
    tree_id: np.ndarray
    site_count: np.ndarray
    pair_to_bucket: Any
    key_parent: Any
    key_child: Any
    key_attach: Any
    bucket_count: Any
    bucket_head: Any
    edge_bucket: np.ndarray
    edge_prev: np.ndarray
    edge_next: np.ndarray
    used_epoch: np.ndarray
    epoch: int = 0

    @classmethod
    def from_compact_states(
        cls,
        states: list["_CompactEdgeTree"],
        *,
        label_capacity: int,
    ) -> "NumbaTrainingForest":
        """Flatten initial compact trees into Numba-friendly arrays."""

        require_numba()
        total_nodes = sum(len(state.parent) for state in states)
        parent = np.full(total_nodes, -1, dtype=np.int64)
        first_child = np.full(total_nodes, -1, dtype=np.int64)
        last_child = np.full(total_nodes, -1, dtype=np.int64)
        next_sibling = np.full(total_nodes, -1, dtype=np.int64)
        prev_sibling = np.full(total_nodes, -1, dtype=np.int64)
        label = np.empty(total_nodes, dtype=np.int64)
        time = np.empty(total_nodes, dtype=np.float64)
        attach = np.empty(total_nodes, dtype=np.int64)
        alive = np.empty(total_nodes, dtype=np.bool_)
        tree_id = np.empty(total_nodes, dtype=np.int64)

        offset = 0
        for current_tree, state in enumerate(states):
            n = len(state.parent)
            for local_node in range(n):
                global_node = offset + local_node
                local_parent = state.parent[local_node]
                parent[global_node] = -1 if local_parent == -1 else offset + local_parent
                label[global_node] = state.label[local_node]
                time[global_node] = state.time[local_node]
                attach[global_node] = state.attach_to_parent[local_node]
                alive[global_node] = state.alive[local_node]
                tree_id[global_node] = current_tree

                previous = -1
                for local_child in state.children[local_node]:
                    global_child = offset + local_child
                    if previous == -1:
                        first_child[global_node] = global_child
                    else:
                        next_sibling[previous] = global_child
                        prev_sibling[global_child] = previous
                    previous = global_child
                last_child[global_node] = previous
            offset += n

        site_count = np.zeros(label_capacity, dtype=np.int64)
        if label.size:
            site_count[: int(label.max()) + 1] = 1
        index = _build_pair_index(parent, label, attach, alive)
        return cls(
            parent=parent,
            first_child=first_child,
            last_child=last_child,
            next_sibling=next_sibling,
            prev_sibling=prev_sibling,
            label=label,
            time=time,
            attach=attach,
            alive=alive,
            tree_id=tree_id,
            site_count=site_count,
            pair_to_bucket=index[0],
            key_parent=index[1],
            key_child=index[2],
            key_attach=index[3],
            bucket_count=index[4],
            bucket_head=index[5],
            edge_bucket=index[6],
            edge_prev=index[7],
            edge_next=index[8],
            used_epoch=np.zeros(total_nodes, dtype=np.int64),
        )

    def select_best_pair(
        self,
        minimum: int,
        codec: "_TokenCodec",
    ) -> tuple["EdgeKey", int] | None:
        """Select with the Python backend's exact deterministic tie policy."""

        best_count, rows = _max_count_buckets(
            minimum,
            self.key_parent,
            self.key_child,
            self.key_attach,
            self.bucket_count,
        )
        if best_count == 0:
            return None
        best_key: tuple[int, int, int] | None = None
        best_priority: tuple[int, str, str] | None = None
        for row in rows:
            parent_id = int(row[1])
            child_id = int(row[2])
            attach_site = int(row[3])
            priority = (
                -attach_site,
                codec.sort_key(parent_id),
                codec.sort_key(child_id),
            )
            if best_priority is None or priority > best_priority:
                best_key = (parent_id, child_id, attach_site)
                best_priority = priority
        if best_key is None:  # pragma: no cover - defensive
            return None
        return best_key, int(best_count)

    def register_label(self, label_id: int, expanded_site_count: int) -> None:
        """Register the site count of a newly interned edge token."""

        if label_id < 0 or label_id >= self.site_count.size:
            raise RuntimeError("Numba label-capacity estimate was too small")
        self.site_count[label_id] = expanded_site_count

    def contract_pair(self, key: "EdgeKey", *, new_label: int) -> int:
        """Contract the same deterministic occurrence snapshot as Python."""

        typed_key = (np.int64(key[0]), np.int64(key[1]), np.int64(key[2]))
        if typed_key not in self.pair_to_bucket:
            return 0
        bucket = int(self.pair_to_bucket[typed_key])
        if self.bucket_count[bucket] == 0:
            return 0
        candidates = _bucket_occurrences(
            bucket,
            self.bucket_count,
            self.bucket_head,
            self.edge_next,
        )
        candidate_parents = self.parent[candidates]
        order = np.lexsort(
            (
                candidates,
                candidate_parents,
                self.time[candidate_parents],
                self.time[candidates],
                self.tree_id[candidates],
            )
        )
        ordered = np.ascontiguousarray(candidates[order], dtype=np.int64)
        self.epoch += 1
        return int(
            _contract_candidates(
                ordered,
                key[0],
                key[1],
                key[2],
                new_label,
                self.epoch,
                self.used_epoch,
                self.parent,
                self.first_child,
                self.last_child,
                self.next_sibling,
                self.prev_sibling,
                self.label,
                self.time,
                self.attach,
                self.alive,
                self.site_count,
                self.pair_to_bucket,
                self.key_parent,
                self.key_child,
                self.key_attach,
                self.bucket_count,
                self.bucket_head,
                self.edge_bucket,
                self.edge_prev,
                self.edge_next,
            )
        )

    def assert_counts_match_recount(self) -> None:
        """Debug helper comparing the incremental index with a full recount."""

        recount = _full_recount(self.parent, self.label, self.attach, self.alive)
        incremental: dict[tuple[int, int, int], int] = {}
        for bucket in range(len(self.bucket_count)):
            count = int(self.bucket_count[bucket])
            if count:
                key = (
                    int(self.key_parent[bucket]),
                    int(self.key_child[bucket]),
                    int(self.key_attach[bucket]),
                )
                incremental[key] = count
        rebuilt = {tuple(int(x) for x in key): int(value) for key, value in recount.items()}
        if incremental != rebuilt:
            raise AssertionError(
                f"incremental Numba pair counts differ from recount: "
                f"incremental={incremental!r}, recount={rebuilt!r}"
            )
