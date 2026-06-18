"""Encoder artifact interfaces and rule metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .vocabulary import Operation, Token, Vocabulary


@dataclass(frozen=True)
class EncodingRule:
    """Model-independent metadata for one learned contraction rule."""

    token: Token
    operation: Operation
    created_at_step: int
    pattern: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeEncoder(ABC):
    """Abstract encoder artifact produced by ``TreeCoarsener.fit``."""

    model_id: str
    vocab: Vocabulary
    rules: Sequence[EncodingRule] = field(default_factory=tuple)
    base_labels: frozenset[str] = field(default_factory=frozenset)

    label_attr: str = "label"
    time_attr: str = "time"
    uid_attr: str = "uid"
    super_uid_attr: str = "super_uids"
    attach_attr: str = "attach_map"

    @abstractmethod
    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        """Encode one directed rooted tree and return a new NetworkX graph."""


@dataclass
class LazyTreeEncoder(TreeEncoder):
    """Lazy composition of fitted encoders in application order."""

    encoders: tuple[TreeEncoder, ...] = ()

    def encode(self, G: nx.DiGraph, *, validate: bool = True) -> nx.DiGraph:
        H = G
        for encoder in self.encoders:
            H = encoder.encode(H, validate=validate)
        return H
