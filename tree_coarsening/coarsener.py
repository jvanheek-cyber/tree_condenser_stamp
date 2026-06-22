"""User-facing abstract coarsener base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Hashable, Literal
from uuid import uuid4

import networkx as nx

from .decoder import TreeDecoder
from .encoder import TreeEncoder
from .exceptions import NotFittedError
from .vocabulary import Token

DecodeBy = Literal["node", "label", "type"]


class TreeCoarsener(ABC):
    """Base class for fitted tree coarseners.

    Subclasses implement ``_fit``. The public ``fit`` method stores the produced
    encoder and decoder in ``encoder_`` and ``decoder_`` and returns ``self``.
    """

    encoder_: TreeEncoder | None
    decoder_: TreeDecoder | None

    def __init__(
        self,
        *,
        label_attr: str = "label",
        time_attr: str = "time",
        uid_attr: str = "uid",
        super_uid_attr: str = "super_uids",
        attach_attr: str = "attach_map",
        validate_inputs: bool = True,
        model_id: str | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        if deprecated_kwargs:
            unknown = ", ".join(sorted(deprecated_kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        self.label_attr = label_attr
        self.time_attr = time_attr
        self.uid_attr = uid_attr
        self.super_uid_attr = super_uid_attr
        self.attach_attr = attach_attr
        self.validate_inputs = validate_inputs
        self.model_id = model_id or f"{self.__class__.__name__}:{uuid4().hex}"
        self.encoder_ = None
        self.decoder_ = None

    def fit(self, graphs: Sequence[nx.DiGraph]) -> "TreeCoarsener":
        """Fit on a nonempty sequence of directed rooted trees."""

        graphs = list(graphs)
        if len(graphs) == 0:
            raise ValueError("fit requires at least one graph.")
        encoder, decoder = self._fit(graphs)
        self.encoder_ = encoder
        self.decoder_ = decoder
        return self

    def transform(self, graph: nx.DiGraph, *, validate: bool = True, **kwargs) -> nx.DiGraph:
        """Encode one graph using the fitted encoder."""

        if self.encoder_ is None:
            raise NotFittedError("Call fit before transform.")
        return self.encoder_.encode(graph, validate=validate, **kwargs)

    def fit_transform(
        self, graphs: Sequence[nx.DiGraph], *, validate: bool = True
    ) -> list[nx.DiGraph]:
        """Fit on ``graphs`` and return their encoded forms."""

        self.fit(graphs)
        return [self.transform(G, validate=validate) for G in graphs]

    def decode(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph:
        """Decode using the fitted decoder."""

        if self.decoder_ is None:
            raise NotFittedError("Call fit before decode.")
        return self.decoder_.decode(
            graph,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def inverse_transform(
        self,
        graph: nx.DiGraph,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph:
        """Alias for ``decode`` using transformer-style naming."""

        return self.decode(
            graph,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def fit_artifacts(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        """Fit and return ``(encoder, decoder)`` directly."""

        self.fit(graphs)
        assert self.encoder_ is not None and self.decoder_ is not None
        return self.encoder_, self.decoder_

    @abstractmethod
    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        """Subclass implementation for fitting."""
