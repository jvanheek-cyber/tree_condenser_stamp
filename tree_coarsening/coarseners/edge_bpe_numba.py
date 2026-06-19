"""Optional Numba fitting backend for attachment-independent edge BPE.

The public graph and decoder semantics remain ordinary Python.  This module
accelerates only the fit-time mutable forest and incremental raw label-pair
counts.  Its pair key is exactly ``(parent_label_id, child_label_id)``; edge
attachment maps are intentionally absent from the fitting state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .edge_bpe import EdgeKey, _CompactEdgeTree, _PairSelection, _TokenCodec

try:  # pragma: no cover - availability depends on the environment
    import numpy as np
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
    """Return whether the optional compiled backend can be used."""

    return _NUMBA_IMPORT_ERROR is None


def require_numba() -> None:
    """Raise an informative error when the optional dependency is absent."""

    if _NUMBA_IMPORT_ERROR is not None:
        raise ImportError(
            "backend='numba' requires the optional dependency. Install "
            "tree-coarsening[numba] or install numba directly."
        ) from _NUMBA_IMPORT_ERROR


if _NUMBA_IMPORT_ERROR is None:
    _PAIR_KEY_TYPE = types.UniTuple(types.int64, 2)

    @njit(cache=True)
    def _add_edge(
        child: int,
        parent: np.ndarray,
        label: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> None:
        p = parent[child]
        key = (label[p], label[child])
        if key in pair_to_bucket:
            bucket = pair_to_bucket[key]
        else:
            bucket = len(bucket_count)
            pair_to_bucket[key] = bucket
            key_parent.append(key[0])
            key_child.append(key[1])
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
        alive: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        pair_to_bucket = Dict.empty(key_type=_PAIR_KEY_TYPE, value_type=types.int64)
        key_parent = List.empty_list(types.int64)
        key_child = List.empty_list(types.int64)
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
                    pair_to_bucket,
                    key_parent,
                    key_child,
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
            bucket_count,
            bucket_head,
            edge_bucket,
            edge_prev,
            edge_next,
        )

    @njit(cache=True)
    def _pair_score(
        score_mode: int,
        n_ab: int,
        n_a: int,
        n_b: int,
        s_a: int,
        s_b: int,
    ) -> float:
        if score_mode == 0:
            return float(n_ab)
        if score_mode == 1:
            if n_a <= 0 or n_b <= 0:
                raise RuntimeError("normalized pair score received nonpositive label count")
            return float(n_ab) / np.sqrt(float(n_a) * float(n_b))
        if score_mode == 2:
            return float(n_ab) * float(s_a + s_b)
        raise RuntimeError("unknown pair-score mode")

    @njit(cache=True)
    def _best_score_buckets(
        minimum: int,
        score_mode: int,
        key_parent: Any,
        key_child: Any,
        bucket_count: Any,
        label_count: np.ndarray,
        label_size: np.ndarray,
    ) -> tuple[float, int, np.ndarray]:
        best_score = -np.inf
        best_count = -1
        for bucket in range(len(bucket_count)):
            count = bucket_count[bucket]
            if count < minimum:
                continue
            parent_id = key_parent[bucket]
            child_id = key_child[bucket]
            score = _pair_score(
                score_mode,
                count,
                label_count[parent_id],
                label_count[child_id],
                label_size[parent_id],
                label_size[child_id],
            )
            if score > best_score or (score == best_score and count > best_count):
                best_score = score
                best_count = count
        if best_count < 0:
            return -np.inf, 0, np.empty((0, 3), dtype=np.int64)

        n_best = 0
        for bucket in range(len(bucket_count)):
            count = bucket_count[bucket]
            if count != best_count:
                continue
            parent_id = key_parent[bucket]
            child_id = key_child[bucket]
            score = _pair_score(
                score_mode,
                count,
                label_count[parent_id],
                label_count[child_id],
                label_size[parent_id],
                label_size[child_id],
            )
            if score == best_score:
                n_best += 1
        rows = np.empty((n_best, 3), dtype=np.int64)
        cursor = 0
        for bucket in range(len(bucket_count)):
            count = bucket_count[bucket]
            if count != best_count:
                continue
            parent_id = key_parent[bucket]
            child_id = key_child[bucket]
            score = _pair_score(
                score_mode,
                count,
                label_count[parent_id],
                label_count[child_id],
                label_size[parent_id],
                label_size[child_id],
            )
            if score == best_score:
                rows[cursor, 0] = bucket
                rows[cursor, 1] = parent_id
                rows[cursor, 2] = child_id
                cursor += 1
        return best_score, best_count, rows

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
        size: np.ndarray,
        time: np.ndarray,
        alive: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
        bucket_count: Any,
        bucket_head: Any,
        edge_bucket: np.ndarray,
        edge_prev: np.ndarray,
        edge_next: np.ndarray,
    ) -> None:
        grandparent = parent[parent_node]

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
            current = next_sibling[current]

        # Preserve Python ordering: surviving parent children first, followed by
        # the removed child's children.
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
        size[parent_node] += size[child_node]
        if time[child_node] > time[parent_node]:
            time[parent_node] = time[child_node]

        alive[child_node] = False
        parent[child_node] = -1
        first_child[child_node] = -1
        last_child[child_node] = -1

        if grandparent != -1:
            _add_edge(
                parent_node,
                parent,
                label,
                pair_to_bucket,
                key_parent,
                key_child,
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
                pair_to_bucket,
                key_parent,
                key_child,
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
        new_label: int,
        epoch: int,
        used_epoch: np.ndarray,
        parent: np.ndarray,
        first_child: np.ndarray,
        last_child: np.ndarray,
        next_sibling: np.ndarray,
        prev_sibling: np.ndarray,
        label: np.ndarray,
        size: np.ndarray,
        time: np.ndarray,
        alive: np.ndarray,
        pair_to_bucket: Any,
        key_parent: Any,
        key_child: Any,
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
            if label[p] != selected_parent_label or label[child] != selected_child_label:
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
                size,
                time,
                alive,
                pair_to_bucket,
                key_parent,
                key_child,
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
        alive: np.ndarray,
    ) -> Any:
        recount = Dict.empty(key_type=_PAIR_KEY_TYPE, value_type=types.int64)
        for child in range(parent.size):
            p = parent[child]
            if alive[child] and p >= 0 and alive[p]:
                key = (label[p], label[child])
                recount[key] = recount.get(key, 0) + 1
        return recount


@dataclass(slots=True)
class NumbaTrainingForest:
    """One flattened fit forest plus a compiled incremental label-pair index."""

    parent: np.ndarray
    first_child: np.ndarray
    last_child: np.ndarray
    next_sibling: np.ndarray
    prev_sibling: np.ndarray
    label: np.ndarray
    size: np.ndarray
    label_count: np.ndarray
    label_size: np.ndarray
    time: np.ndarray
    alive: np.ndarray
    tree_id: np.ndarray
    pair_to_bucket: Any
    key_parent: Any
    key_child: Any
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
        """Flatten initial Python fit states into Numba-friendly arrays."""

        require_numba()
        total_nodes = sum(len(state.parent) for state in states)
        parent = np.full(total_nodes, -1, dtype=np.int64)
        first_child = np.full(total_nodes, -1, dtype=np.int64)
        last_child = np.full(total_nodes, -1, dtype=np.int64)
        next_sibling = np.full(total_nodes, -1, dtype=np.int64)
        prev_sibling = np.full(total_nodes, -1, dtype=np.int64)
        label = np.empty(total_nodes, dtype=np.int64)
        size = np.empty(total_nodes, dtype=np.int64)
        label_count = np.zeros(label_capacity, dtype=np.int64)
        label_size = np.zeros(label_capacity, dtype=np.int64)
        time = np.empty(total_nodes, dtype=np.float64)
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
                size[global_node] = state.size[local_node]
                label_id = int(label[global_node])
                if label_id >= label_capacity:
                    raise RuntimeError("Numba label capacity is too small")
                label_count[label_id] += 1
                if label_size[label_id] == 0:
                    label_size[label_id] = size[global_node]
                elif label_size[label_id] != size[global_node]:
                    raise RuntimeError("one fitting label has inconsistent sizes")
                time[global_node] = state.time[local_node]
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

        index = _build_pair_index(parent, label, alive)
        return cls(
            parent=parent,
            first_child=first_child,
            last_child=last_child,
            next_sibling=next_sibling,
            prev_sibling=prev_sibling,
            label=label,
            size=size,
            label_count=label_count,
            label_size=label_size,
            time=time,
            alive=alive,
            tree_id=tree_id,
            pair_to_bucket=index[0],
            key_parent=index[1],
            key_child=index[2],
            bucket_count=index[3],
            bucket_head=index[4],
            edge_bucket=index[5],
            edge_prev=index[6],
            edge_next=index[7],
            used_epoch=np.zeros(total_nodes, dtype=np.int64),
        )

    def select_best_pair(
        self,
        minimum: int,
        codec: "_TokenCodec",
        *,
        score_mode: int,
    ) -> "_PairSelection | None":
        """Select with the Python backend's exact deterministic tie policy."""

        best_score, best_count, rows = _best_score_buckets(
            minimum,
            score_mode,
            self.key_parent,
            self.key_child,
            self.bucket_count,
            self.label_count,
            self.label_size,
        )
        if best_count == 0:
            return None
        best_key: tuple[int, int] | None = None
        best_priority: tuple[str, str] | None = None
        for row in rows:
            parent_id = int(row[1])
            child_id = int(row[2])
            priority = (codec.sort_key(parent_id), codec.sort_key(child_id))
            if best_priority is None or priority > best_priority:
                best_key = (parent_id, child_id)
                best_priority = priority
        if best_key is None:  # pragma: no cover - defensive
            return None
        from .edge_bpe import _PairSelection

        parent_id, child_id = best_key
        return _PairSelection(
            key=best_key,
            count=int(best_count),
            parent_count=int(self.label_count[parent_id]),
            child_count=int(self.label_count[child_id]),
            parent_size=int(self.label_size[parent_id]),
            child_size=int(self.label_size[child_id]),
            score=float(best_score),
        )

    def register_label(self, label_id: int, *, size: int) -> None:
        """Register a newly learned fitting label before its first contraction."""

        if label_id < 0 or label_id >= self.label_count.size:
            raise RuntimeError("Numba label capacity is too small")
        self.label_count[label_id] = 0
        self.label_size[label_id] = int(size)

    def contract_pair(self, key: "EdgeKey", *, new_label: int) -> int:
        """Contract the same deterministic occurrence snapshot as Python."""

        typed_key = (np.int64(key[0]), np.int64(key[1]))
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
        events = int(
            _contract_candidates(
                ordered,
                key[0],
                key[1],
                new_label,
                self.epoch,
                self.used_epoch,
                self.parent,
                self.first_child,
                self.last_child,
                self.next_sibling,
                self.prev_sibling,
                self.label,
                self.size,
                self.time,
                self.alive,
                self.pair_to_bucket,
                self.key_parent,
                self.key_child,
                self.bucket_count,
                self.bucket_head,
                self.edge_bucket,
                self.edge_prev,
                self.edge_next,
            )
        )
        if events:
            parent_id, child_id = key
            if parent_id == child_id:
                self.label_count[parent_id] -= 2 * events
            else:
                self.label_count[parent_id] -= events
                self.label_count[child_id] -= events
            self.label_count[new_label] += events
            if self.label_count[parent_id] < 0 or self.label_count[child_id] < 0:
                raise RuntimeError("incremental label occurrence count became negative")
        return events

    def assert_counts_match_recount(self) -> None:
        """Debug helper comparing the incremental index with a full recount."""

        recount = _full_recount(self.parent, self.label, self.alive)
        incremental: dict[tuple[int, int], int] = {}
        for bucket in range(len(self.bucket_count)):
            count = int(self.bucket_count[bucket])
            if count:
                key = (int(self.key_parent[bucket]), int(self.key_child[bucket]))
                incremental[key] = count
        rebuilt = {tuple(int(x) for x in key): int(value) for key, value in recount.items()}
        if incremental != rebuilt:
            raise AssertionError(
                "incremental Numba pair counts differ from recount: "
                f"incremental={incremental!r}, recount={rebuilt!r}"
            )
