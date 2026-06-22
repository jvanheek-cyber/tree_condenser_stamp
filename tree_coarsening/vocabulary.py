"""Vocabulary objects and token utilities for staged tree coarsening."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable as HashableABC, Mapping
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, TypeAlias

from .exceptions import ValidationError

Token: TypeAlias = Hashable
AttachMap: TypeAlias = tuple[int, ...]
Operation: TypeAlias = Literal["base", "edge", "siblings", "component"]

BASE_NAMESPACE = "base"


@dataclass(frozen=True, slots=True)
class TokenSpec:
    """Fixed expanded size/root metadata for an opaque fitting symbol."""

    site_count: int
    root_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.site_count, int)
            or isinstance(self.site_count, bool)
            or self.site_count <= 0
        ):
            raise ValidationError(
                f"TokenSpec.site_count must be positive; got {self.site_count!r}."
            )
        if (
            not isinstance(self.root_count, int)
            or isinstance(self.root_count, bool)
            or self.root_count <= 0
        ):
            raise ValidationError(
                f"TokenSpec.root_count must be positive; got {self.root_count!r}."
            )



def base_token(raw_label: str) -> tuple[str, str]:
    """Return the base-token id for a raw string label."""

    if not isinstance(raw_label, str):
        raise ValidationError(f"raw labels must be strings; got {raw_label!r}.")
    return (BASE_NAMESPACE, raw_label)


def is_base_token(token: Any) -> bool:
    """Return whether ``token`` is a base-token id of the form ``('base', label)``."""

    return (
        isinstance(token, tuple)
        and len(token) == 2
        and token[0] == BASE_NAMESPACE
        and isinstance(token[1], str)
    )


def raw_label_from_base_token(token: Any) -> str:
    """Return the raw label carried by a base token."""

    if not is_base_token(token):
        raise ValidationError(f"not a base token: {token!r}.")
    return token[1]


def format_token(token: Token) -> str:
    """Human-readable display string for common token ids."""

    if is_base_token(token):
        return f"base:{token[1]}"
    if isinstance(token, tuple) and len(token) >= 1:
        return ":".join(str(x) for x in token)
    return str(token)


def normalize_attach_map(value: Any) -> AttachMap:
    """Normalize a scalar or sequence into a tuple-valued attachment map."""

    if isinstance(value, bool):
        raise ValidationError("attachment sites must be integers, not booleans.")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple):
        out = value
    elif isinstance(value, list):
        out = tuple(value)
    else:
        raise ValidationError(f"attach_map must be an int or a sequence of ints; got {value!r}.")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in out):
        raise ValidationError(f"attach_map entries must be integers; got {value!r}.")
    return tuple(int(x) for x in out)


@dataclass(frozen=True, slots=True)
class VocabEntry:
    """One staged canonical vocabulary recipe.

    ``parent`` and ``label`` are position-aligned. ``attach`` is a flat vector of
    integer attachment sites, parsed in recipe order over the non-root
    positions.
    """

    token: Token
    parent: tuple[int, ...]
    label: tuple[Token, ...]
    attach: tuple[int, ...]
    created_at_step: int
    operation: Operation
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.token, HashableABC):
            raise ValidationError(f"token id must be hashable; got {self.token!r}.")
        if is_base_token(self.token):
            raise ValidationError("base tokens are implicit and must not be learned entries.")
        if len(self.parent) != len(self.label):
            raise ValidationError("parent and label must have the same length.")
        if len(self.parent) == 0:
            raise ValidationError("vocabulary entries must contain at least one recipe component.")
        for i, p in enumerate(self.parent):
            if not isinstance(p, int) or isinstance(p, bool):
                raise ValidationError(f"parent[{i}] must be an integer; got {p!r}.")
            if p < -1 or p >= len(self.parent):
                raise ValidationError(f"parent[{i}]={p!r} is outside the allowed range.")
            if p == i:
                raise ValidationError(f"parent[{i}] cannot point to itself.")
        for i, token in enumerate(self.label):
            if not isinstance(token, HashableABC):
                raise ValidationError(f"label[{i}] must be a hashable token id; got {token!r}.")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in self.attach):
            raise ValidationError("attach must be a flat tuple of integers.")
        if self.operation not in {"edge", "siblings", "component", "base"}:
            raise ValidationError(f"unknown operation {self.operation!r}.")
        self._check_acyclic_parent_relation()

    @property
    def n_components(self) -> int:
        return len(self.parent)

    @property
    def root_positions(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.parent) if p == -1)

    def attachment_slice(
        self,
        i: int,
        vocab: Mapping[Token, "VocabEntry"] | "Vocabulary",
    ) -> AttachMap:
        """Return the slice of ``A`` associated with recipe position ``i``."""

        if i < 0 or i >= self.n_components:
            raise IndexError(i)
        if self.parent[i] == -1:
            return ()
        vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
        start = sum(
            vocabulary.root_count(self.label[h])
            for h in range(i)
            if self.parent[h] >= 0
        )
        stop = start + vocabulary.root_count(self.label[i])
        return tuple(self.attach[start:stop])

    def attachment_slices(
        self, vocab: Mapping[Token, "VocabEntry"] | "Vocabulary"
    ) -> tuple[AttachMap, ...]:
        """Return all position-aligned attachment slices in one linear pass."""

        vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
        out: list[AttachMap] = []
        cursor = 0
        for i, parent in enumerate(self.parent):
            if parent == -1:
                out.append(())
                continue
            width = vocabulary.root_count(self.label[i])
            out.append(tuple(self.attach[cursor : cursor + width]))
            cursor += width
        if cursor != len(self.attach):
            raise ValidationError(
                f"entry {self.token!r} consumed {cursor} attachment values, "
                f"but stores {len(self.attach)}."
            )
        return tuple(out)

    def _check_acyclic_parent_relation(self) -> None:
        """Check a parent-pointer forest in linear time."""

        # 0 = unseen, 1 = on the current parent chain, 2 = fully checked.
        state = bytearray(len(self.parent))
        for start in range(len(self.parent)):
            if state[start] == 2:
                continue
            path: list[int] = []
            cur = start
            while cur != -1 and state[cur] == 0:
                state[cur] = 1
                path.append(cur)
                cur = self.parent[cur]
            if cur != -1 and state[cur] == 1:
                raise ValidationError("recipe parent relation is cyclic.")
            for node in path:
                state[node] = 2


@dataclass
class Vocabulary:
    """Temporal staged vocabulary with cached root and site counts.

    Base tokens are implicit and have one exposed root and one site. Counts for
    learned tokens are computed once when entries are added. Constructing a
    vocabulary from an existing mapping rebuilds the same caches iteratively, so
    deeply nested vocabularies do not depend on Python recursion depth.
    """

    entries: dict[Token, VocabEntry] = field(default_factory=dict)
    creation_order: list[Token] = field(default_factory=list)
    symbols: dict[Token, TokenSpec] = field(default_factory=dict)
    _root_counts: dict[Token, int] = field(default_factory=dict, init=False, repr=False)
    _site_counts: dict[Token, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.symbols = dict(self.symbols)
        overlap = set(self.entries) & set(self.symbols)
        if overlap:
            raise ValidationError(
                f"tokens cannot be both static entries and opaque symbols: "
                f"{sorted(overlap, key=repr)!r}."
            )
        if not self.creation_order:
            self.creation_order = list(self.entries)
        else:
            unknown = [token for token in self.creation_order if token not in self.entries]
            if unknown:
                raise ValidationError(
                    f"creation_order contains tokens missing from entries: {unknown!r}."
                )
            for token in self.entries:
                if token not in self.creation_order:
                    self.creation_order.append(token)
        if self.entries:
            self._rebuild_count_caches()

    def __contains__(self, token: Token) -> bool:
        return is_base_token(token) or token in self.entries or token in self.symbols

    def __getitem__(self, token: Token) -> VocabEntry:
        return self.entries[token]

    def get(self, token: Token, default: Any = None) -> VocabEntry | Any:
        return self.entries.get(token, default)

    def items(self):
        return self.entries.items()

    def as_mapping(self) -> dict[Token, VocabEntry]:
        return dict(self.entries)

    def add_symbol(self, token: Token, spec: TokenSpec) -> None:
        """Register an opaque fitting symbol with fixed coordinate counts."""

        if not isinstance(token, HashableABC):
            raise ValidationError(f"symbol token must be hashable; got {token!r}.")
        if token in self.entries:
            raise ValidationError(f"token {token!r} is already a static vocabulary entry.")
        previous = self.symbols.get(token)
        if previous is not None and previous != spec:
            raise ValidationError(
                f"symbol {token!r} already has specification {previous!r}, not {spec!r}."
            )
        self.symbols[token] = spec

    def add(self, entry: VocabEntry) -> None:
        """Add and validate one learned token, caching its counts immediately."""

        if entry.token in self.entries:
            raise ValidationError(f"duplicate vocabulary token {entry.token!r}.")
        if entry.token in self.symbols:
            raise ValidationError(
                f"token {entry.token!r} is already registered as an opaque symbol."
            )
        self.validate_entry(entry)
        site_count_value = sum(self.site_count(token) for token in entry.label)
        root_count_value = sum(
            self.root_count(entry.label[i]) for i in entry.root_positions
        )
        self.entries[entry.token] = entry
        self.creation_order.append(entry.token)
        self._site_counts[entry.token] = site_count_value
        self._root_counts[entry.token] = root_count_value

    def remove_last(self, token: Token) -> None:
        """Remove ``token`` when it is the most recently added learned entry.

        This is intended only for defensive rollback inside fitting code.
        """

        if not self.creation_order or self.creation_order[-1] != token:
            raise ValidationError(f"token {token!r} is not the last vocabulary entry.")
        self.creation_order.pop()
        self.entries.pop(token, None)
        self._root_counts.pop(token, None)
        self._site_counts.pop(token, None)

    def root_count(self, token: Token) -> int:
        """Number of exposed roots of ``token`` after full expansion."""

        if is_base_token(token):
            return 1
        if token in self.symbols:
            return self.symbols[token].root_count
        try:
            return self._root_counts[token]
        except KeyError as exc:
            if token not in self.entries:
                raise ValidationError(f"unknown token {token!r}.") from exc
            self._rebuild_count_caches()
            return self._root_counts[token]

    def site_count(self, token: Token) -> int:
        """Number of expanded vertices/sites of ``token``."""

        if is_base_token(token):
            return 1
        if token in self.symbols:
            return self.symbols[token].site_count
        try:
            return self._site_counts[token]
        except KeyError as exc:
            if token not in self.entries:
                raise ValidationError(f"unknown token {token!r}.") from exc
            self._rebuild_count_caches()
            return self._site_counts[token]

    def validate_entry(self, entry: VocabEntry) -> None:
        """Validate dependencies, attachment lengths, and attachment ranges."""

        for i, token in enumerate(entry.label):
            if (
                not is_base_token(token)
                and token not in self.entries
                and token not in self.symbols
            ):
                raise ValidationError(
                    f"entry {entry.token!r} references unknown or future token {token!r} "
                    f"at label[{i}]."
                )

        expected_len = sum(
            self.root_count(entry.label[i]) for i, p in enumerate(entry.parent) if p >= 0
        )
        if len(entry.attach) != expected_len:
            raise ValidationError(
                f"entry {entry.token!r} has A length {len(entry.attach)}, expected {expected_len}."
            )

        attachment_slices = entry.attachment_slices(self)
        for i, p in enumerate(entry.parent):
            if p == -1:
                continue
            parent_sites = self.site_count(entry.label[p])
            bad = [q for q in attachment_slices[i] if q < 0 or q >= parent_sites]
            if bad:
                raise ValidationError(
                    f"entry {entry.token!r} has attachment sites {bad!r} outside "
                    f"0..{parent_sites - 1} for position {i}."
                )

    def _rebuild_count_caches(self) -> None:
        """Rebuild count caches with an iterative dependency topological pass."""

        self._root_counts.clear()
        self._site_counts.clear()

        dependents: dict[Token, list[Token]] = defaultdict(list)
        indegree: dict[Token, int] = {}

        for token, entry in self.entries.items():
            deps = {
                child
                for child in entry.label
                if not is_base_token(child) and child not in self.symbols
            }
            missing = [child for child in deps if child not in self.entries]
            if missing:
                raise ValidationError(
                    f"entry {token!r} references unknown tokens {missing!r}."
                )
            indegree[token] = len(deps)
            for dep in deps:
                dependents[dep].append(token)

        order_rank = {token: i for i, token in enumerate(self.creation_order)}
        ready = deque(
            sorted(
                (token for token, degree in indegree.items() if degree == 0),
                key=lambda token: order_rank.get(token, len(order_rank)),
            )
        )
        processed = 0

        while ready:
            token = ready.popleft()
            entry = self.entries[token]
            self._site_counts[token] = sum(
                (
                    1
                    if is_base_token(child)
                    else self.symbols[child].site_count
                    if child in self.symbols
                    else self._site_counts[child]
                )
                for child in entry.label
            )
            self._root_counts[token] = sum(
                (
                    1
                    if is_base_token(entry.label[i])
                    else self.symbols[entry.label[i]].root_count
                    if entry.label[i] in self.symbols
                    else self._root_counts[entry.label[i]]
                )
                for i in entry.root_positions
            )
            processed += 1
            for dependent in dependents.get(token, ()):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if processed != len(self.entries):
            cyclic = [token for token, degree in indegree.items() if degree > 0]
            raise ValidationError(
                f"vocabulary token dependencies are cyclic: {cyclic!r}."
            )


def root_count(token: Token, vocab: Mapping[Token, VocabEntry] | Vocabulary) -> int:
    vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
    return vocabulary.root_count(token)


def site_count(token: Token, vocab: Mapping[Token, VocabEntry] | Vocabulary) -> int:
    vocabulary = vocab if isinstance(vocab, Vocabulary) else Vocabulary(entries=dict(vocab))
    return vocabulary.site_count(token)
