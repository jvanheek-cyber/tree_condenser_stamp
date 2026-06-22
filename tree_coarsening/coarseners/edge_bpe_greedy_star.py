"""Greedy star-first edge BPE for directed labeled trees.

This is an ordinary edge-BPE learner with a single twist in *rule selection*.
After the global best ``(parent_label, child_label)`` pair is merged into a new
token ``T``, fitting does **not** immediately return to the global frequency
ranking.  Instead it greedily prefers extending ``T`` with another child that is
structurally one of ``T``'s own merged components -- the original repeated child
*or* a sibling that has already been condensed into a component token -- as long
as at least one such edge remains.  ``min_pair_count`` gates only the global pick
that *starts* a chain; greedy continuations ignore it, so a star is consumed to
its last child.  Recognizing condensed siblings (not just the raw child label)
ensures a node never keeps a child whose subtree equals a component it already
absorbed, while the learned rules remain an ordinary ordered BPE merge table.

Because the rules are plain edge-BPE rules, encoding new data is exactly
straightforward edge BPE: the fitted :class:`~tree_coarsening.EdgeBPEEncoder`
applies the ordered merge table, and a star is reabsorbed by repeatedly applying
the chained ``(parent, child) -> (T0, child) -> (T1, child) -> ...`` rules.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..decoder import TreeDecoder
from ..encoder import EncodingRule, TreeEncoder
from ..exceptions import ValidationError
from ..schema import RAW_INPUT_FLAG
from ..stage_decoder import StructuralStageDecoder
from ..structural import infer_input_alphabet
from ..validation import validate_coarsenable_tree
from ..vocabulary import TokenSpec, Vocabulary
from .edge_bpe import (
    EdgeBPECoarsener,
    EdgeBPEEncoder,
    EdgeBPERule,
    EdgeKey,
    PairScore,
    _CompactEdgeTree,
    _PairSelection,
    _TokenCodec,
    _initial_label_statistics,
    _set_new_label_statistics,
    _update_label_counts_after_merge,
    edge_bpe_token,
)

__all__ = [
    "GreedyStarBPECoarsener",
    "GreedyStarBPEEncoder",
    "EdgeBPEEncoder",
    "EdgeBPERule",
    "edge_bpe_token",
]


@dataclass
class GreedyStarBPEEncoder(EdgeBPEEncoder):
    """Edge-BPE encoder whose ``max_steps`` counts whole star-chains.

    The greedy learner emits each star as a *chain* of merge rules
    ``(parent, child) -> (T0, child) -> (T1, child) -> ...``.  ``group_starts``
    marks, for every learned rule, whether it begins a new chain (the first rule
    of a chain is the non-greedy global pick; the rest are greedy
    continuations).  ``max_steps`` is interpreted as the number of *complete*
    chains to apply, so encoding never stops on a partially merged star.
    """

    group_starts: tuple[bool, ...] = ()

    def _rule_limit_for_groups(self, max_steps: int | None) -> int | None:
        """Translate a chain count into a rule count for the base encoder."""

        if max_steps is None:
            return None
        if max_steps < 0:
            raise ValueError("max_steps must be None or nonnegative.")
        # Without alignment information fall back to per-rule semantics.
        if len(self.group_starts) != len(self.edge_rules):
            return max_steps
        groups_seen = 0
        for i, is_start in enumerate(self.group_starts):
            if is_start:
                if groups_seen == max_steps:
                    return i
                groups_seen += 1
        return None

    def encode(
        self, G: nx.DiGraph, *, validate: bool = True, max_steps: int = None
    ) -> nx.DiGraph:
        return super().encode(
            G,
            validate=validate,
            max_steps=self._rule_limit_for_groups(max_steps),
        )


class GreedyStarBPECoarsener(EdgeBPECoarsener):
    """Edge BPE that greedily chains ``(new_token, same_child)`` merges.

    The learned model is an ordinary edge-BPE merge table, so the fitted encoder
    and decoder are identical to :class:`~tree_coarsening.EdgeBPECoarsener`.  The
    only difference is the order in which pairs are selected during fitting:
    after a merge the coarsener prefers extending the freshly created token with
    another child that is structurally one of the token's own merged components
    (its merge-closure), contracting a star of identical children one sibling at
    a time before returning to the global frequency ranking.  Because condensed
    siblings are matched by their component label -- not just the raw child label
    -- a node never keeps a child equal to a component it already absorbed.
    ``min_pair_count`` gates only the global pick that starts a chain; greedy
    continuations ignore it and consume the star down to its last child.

    Fitting is Python-only; the chained greedy selection is not implemented for
    the Numba backend.
    """

    def __init__(
        self,
        *,
        num_merges: int | None = None,
        min_pair_count: int = 2,
        pair_score: PairScore = "count",
        **kwargs: Any,
    ) -> None:
        backend = kwargs.pop("backend", "python")
        if backend != "python":
            raise ValueError(
                "GreedyStarBPECoarsener supports backend='python' only."
            )
        super().__init__(
            num_merges=num_merges,
            min_pair_count=min_pair_count,
            pair_score=pair_score,
            backend="python",
            **kwargs,
        )

    def _selection_for_key(
        self,
        key: EdgeKey,
        count: int,
        codec: _TokenCodec,
        label_counts: Sequence[int],
        label_sizes: Sequence[int],
    ) -> _PairSelection:
        """Score one specific pair (used for greedy continuation)."""

        parent_id, child_id = key
        parent_count = int(label_counts[parent_id])
        child_count = int(label_counts[child_id])
        parent_size = int(label_sizes[parent_id])
        child_size = int(label_sizes[child_id])
        try:
            score = float(
                self._pair_score_function(
                    int(count),
                    parent_count,
                    child_count,
                    parent_size,
                    child_size,
                )
            )
        except Exception as exc:  # pragma: no cover - mirrors _select_best_pair
            raise ValidationError(
                f"pair_score failed for pair "
                f"({codec.decode(parent_id)!r}, {codec.decode(child_id)!r})."
            ) from exc
        if not math.isfinite(score):
            raise ValidationError(
                f"pair_score returned non-finite value {score!r} for "
                f"N(A,B)={count}, N(A)={parent_count}, N(B)={child_count}, "
                f"S(A)={parent_size}, S(B)={child_size}."
            )
        return _PairSelection(
            key=key,
            count=int(count),
            parent_count=parent_count,
            child_count=child_count,
            parent_size=parent_size,
            child_size=child_size,
            score=score,
        )

    def _best_greedy_continuation(
        self,
        parent_id: int,
        component_labels: frozenset[int],
        counts: Counter[EdgeKey],
        codec: _TokenCodec,
        label_counts: Sequence[int],
        label_sizes: Sequence[int],
    ) -> _PairSelection | None:
        """Best greedy continuation off the freshly created chain token.

        The chain absorbs any remaining child of ``parent_id`` whose label is
        structurally one of the token's own merged components (``component_labels``
        is the merge-closure of the token).  This includes the original repeated
        child *and* siblings that have already been condensed into a component
        token -- the latter being the case the single-label chain used to strand.
        Among all such candidates the highest-priority pair is chosen, mirroring
        :meth:`_select_best_pair`'s ordering; ``min_pair_count`` does not apply.
        """

        best: _PairSelection | None = None
        best_priority: tuple[float, int, str, str] | None = None
        for child_id in component_labels:
            key = (parent_id, child_id)
            count = counts.get(key, 0)
            if count < 1:
                continue
            selection = self._selection_for_key(
                key, count, codec, label_counts, label_sizes
            )
            priority = (
                selection.score,
                selection.count,
                codec.sort_key(parent_id),
                codec.sort_key(child_id),
            )
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best = selection
        return best

    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        self.backend_used_ = "python"
        input_alphabet = infer_input_alphabet(
            graphs,
            label_attr=self.label_attr,
            type_attr=self.type_attr,
            size_attr=self.size_attr,
            attach_attr=self.attach_attr,
        )
        vocab = Vocabulary(symbols=input_alphabet)
        codec = _TokenCodec()
        counts: Counter[EdgeKey] = Counter()
        states: list[_CompactEdgeTree] = []

        for graph in graphs:
            if self.validate_inputs:
                validate_coarsenable_tree(
                    graph,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                )
            states.append(
                _CompactEdgeTree.from_graph(
                    graph,
                    codec=codec,
                    vocab=vocab,
                    label_attr=self.label_attr,
                    type_attr=self.type_attr,
                    size_attr=self.size_attr,
                    time_attr=self.time_attr,
                    uid_attr=self.uid_attr,
                    super_label_attr=self.super_label_attr,
                    super_uid_attr=self.super_uid_attr,
                    attach_attr=self.attach_attr,
                    pair_counts=counts,
                    capture_output=False,
                    build_edge_index=True,
                )
            )

        label_counts, label_sizes = _initial_label_statistics(states, codec, vocab)

        learned: list[EdgeBPERule] = []
        group_starts: list[bool] = []
        encoding_rules: list[EncodingRule] = []
        self.history_ = []
        rank = 0
        # When set, the freshly created token to keep extending: the chain
        # greedily absorbs any of its children whose label is structurally one of
        # the token's own merged components.  ``component_closure`` maps every
        # learned token to that set of component labels (its merge-closure).
        greedy_parent: int | None = None
        component_closure: dict[int, frozenset[int]] = {}

        while self.num_merges is None or rank < self.num_merges:
            best: _PairSelection | None = None
            is_greedy = False
            if greedy_parent is not None:
                # ``min_pair_count`` gates only the global pick that *starts* a
                # chain; once a star is being consumed keep eating component-equal
                # children as long as at least one remains.
                best = self._best_greedy_continuation(
                    greedy_parent,
                    component_closure.get(greedy_parent, frozenset()),
                    counts,
                    codec,
                    label_counts,
                    label_sizes,
                )
                if best is not None:
                    is_greedy = True
            if best is None:
                best = self._select_best_pair(
                    counts, codec, label_counts, label_sizes
                )
            if best is None:
                break

            key = best.key
            raw_count = best.count
            parent_id, child_id = key
            parent_label = codec.decode(parent_id)
            child_label = codec.decode(child_id)
            token = edge_bpe_token(rank)

            parent_spec = vocab.symbols[parent_label]
            child_spec = vocab.symbols[child_label]
            vocab.add_symbol(
                token,
                TokenSpec(
                    site_count=parent_spec.site_count + child_spec.site_count,
                    root_count=parent_spec.root_count,
                ),
            )
            new_id = codec.intern(token)
            _set_new_label_statistics(
                label_counts,
                label_sizes,
                label_id=new_id,
                size=best.parent_size + best.child_size,
            )
            # Merge-closure of the new token: its two direct operands plus all of
            # their own components.  A child carrying any of these labels is a
            # structural copy of a component, so the greedy chain absorbs it.
            component_closure[new_id] = (
                frozenset((parent_id, child_id))
                | component_closure.get(parent_id, frozenset())
                | component_closure.get(child_id, frozenset())
            )

            actual_events = sum(
                state.contract_pair(key, new_label=new_id, pair_counts=counts)
                for state in states
            )
            _update_label_counts_after_merge(
                label_counts,
                parent_id=parent_id,
                child_id=child_id,
                new_id=new_id,
                events=actual_events,
            )
            if actual_events == 0:
                # A positive incremental count guarantees a contractible edge,
                # so this is unreachable; recover defensively all the same.
                vocab.symbols.pop(token, None)
                counts.pop(key, None)
                component_closure.pop(new_id, None)
                greedy_parent = None
                continue

            rule = EdgeBPERule(
                rank=rank,
                token=token,
                parent_label=parent_label,
                child_label=child_label,
                count=raw_count,
                score=best.score,
                parent_count=best.parent_count,
                child_count=best.child_count,
                parent_size=best.parent_size,
                child_size=best.child_size,
            )
            learned.append(rule)
            # A non-greedy pick begins a new star-chain; greedy picks extend it.
            group_starts.append(not is_greedy)
            encoding_rules.append(
                EncodingRule(
                    token=token,
                    operation="edge",
                    created_at_step=rank,
                    pattern={
                        "parent_label": parent_label,
                        "child_label": child_label,
                    },
                    score=best.score,
                    metadata={
                        "actual_events": actual_events,
                        "count_semantics": "raw_matching_edges",
                        "pair_score": self.pair_score_display_name_,
                        "raw_count": raw_count,
                        "parent_count": best.parent_count,
                        "child_count": best.child_count,
                        "parent_size": best.parent_size,
                        "child_size": best.child_size,
                        "greedy": is_greedy,
                    },
                )
            )
            self.history_.append(
                {
                    "rank": rank,
                    "token": token,
                    "parent_label": parent_label,
                    "child_label": child_label,
                    # Compatibility aliases.
                    "parent_token": parent_label,
                    "child_token": child_label,
                    "count": raw_count,
                    "count_semantics": "raw_matching_edges",
                    "parent_count": best.parent_count,
                    "child_count": best.child_count,
                    "parent_size": best.parent_size,
                    "child_size": best.child_size,
                    "score": best.score,
                    "pair_score": self.pair_score_display_name_,
                    "actual_events": actual_events,
                    "greedy": is_greedy,
                }
            )

            # Continue eating component-equal children off the freshly created
            # token (the original repeated child and any condensed siblings).
            greedy_parent = new_id
            rank += 1

        output_raw = all(graph.graph.get(RAW_INPUT_FLAG, False) for graph in graphs)
        encoder = GreedyStarBPEEncoder(
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
            edge_rules=tuple(learned),
            group_starts=tuple(group_starts),
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
