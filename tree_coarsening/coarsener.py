"""User-facing abstract coarsener base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Hashable, Literal
from uuid import uuid4

import networkx as nx

from .decoder import TreeDecoder
from .encoder import TreeEncoder
from .exceptions import NotFittedError, ValidationError
from .schema import RAW_INPUT_FLAG, GraphInput, as_graph_list, normalize_coarsenable_tree
from .validation import validate_coarsenable_tree
from .vocabulary import Token

DecodeBy = Literal["node", "label", "type"]


class TreeCoarsener(ABC):
    """Base class for fitted tree coarseners.

    ``fit`` and ``transform`` accept either one directed tree or a sequence of
    trees.  Subclasses always receive normalized copies satisfying the common
    fitting contract: hashable ``label``/``type``, positive ``size``, numeric
    ``time``, and provenance fields.  Output shape matches input shape.
    """

    encoder_: TreeEncoder | None
    decoder_: TreeDecoder | None

    def __init__(
        self,
        *,
        label_attr: str = "label",
        type_attr: str = "type",
        size_attr: str = "size",
        time_attr: str = "time",
        uid_attr: str = "uid",
        super_label_attr: str = "super_label",
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
        self.type_attr = type_attr
        self.size_attr = size_attr
        self.time_attr = time_attr
        self.uid_attr = uid_attr
        self.super_label_attr = super_label_attr
        self.super_uid_attr = super_uid_attr
        self.attach_attr = attach_attr
        self.validate_inputs = validate_inputs
        self.model_id = model_id or f"{self.__class__.__name__}:{uuid4().hex}"
        self.encoder_ = None
        self.decoder_ = None

    def _prepare_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        prepared = normalize_coarsenable_tree(
            graph,
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
        if self.validate_inputs:
            validate_coarsenable_tree(
                prepared,
                label_attr=self.label_attr,
                type_attr=self.type_attr,
                size_attr=self.size_attr,
                time_attr=self.time_attr,
                super_label_attr=self.super_label_attr,
                super_uid_attr=self.super_uid_attr,
            )
        return prepared

    def fit(self, graphs: GraphInput) -> "TreeCoarsener":
        """Fit on one tree or a nonempty sequence of trees."""

        graph_list, _ = as_graph_list(graphs)
        prepared = [self._prepare_graph(graph) for graph in graph_list]
        input_stages = {
            bool(graph.graph.get(RAW_INPUT_FLAG, False)) for graph in prepared
        }
        if len(input_stages) != 1:
            raise ValidationError(
                "one fit call cannot mix raw trees with previously transformed "
                "trees; fit each pipeline stage on one common graph schema."
            )
        encoder, decoder = self._fit(prepared)
        self.encoder_ = encoder
        self.decoder_ = decoder
        return self

    def transform(
        self,
        graphs: GraphInput,
        *,
        validate: bool = True,
        **kwargs
    ) -> nx.DiGraph | list[nx.DiGraph]:
        """Encode one tree or a sequence, preserving the input container shape."""

        if self.encoder_ is None:
            raise NotFittedError("Call fit before transform.")
        graph_list, was_single = as_graph_list(graphs, argument_name="graphs")
        outputs = [
            self.encoder_.encode(self._prepare_graph(graph), validate=validate, **kwargs)
            for graph in graph_list
        ]
        return outputs[0] if was_single else outputs

    def fit_transform(
        self,
        graphs: GraphInput,
        *,
        validate: bool = True,
    ) -> nx.DiGraph | list[nx.DiGraph]:
        """Fit and transform one tree or a sequence of trees."""

        self.fit(graphs)
        return self.transform(graphs, validate=validate)

    def decode(
        self,
        graphs: GraphInput,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph | list[nx.DiGraph]:
        """Decode one tree or a sequence, preserving the input container shape."""

        if self.decoder_ is None:
            raise NotFittedError("Call fit before decode.")
        graph_list, was_single = as_graph_list(graphs, argument_name="graphs")
        outputs = [
            self.decoder_.decode(
                graph,
                target=target,
                by=by,
                recursive=recursive,
                boundary_policy=boundary_policy,
                validate=validate,
            )
            for graph in graph_list
        ]
        return outputs[0] if was_single else outputs

    def inverse_transform(
        self,
        graphs: GraphInput,
        *,
        target: Hashable | Token | None = None,
        by: DecodeBy = "node",
        recursive: bool = True,
        boundary_policy: Literal["expand", "raise"] = "expand",
        validate: bool = True,
    ) -> nx.DiGraph | list[nx.DiGraph]:
        """Alias for ``decode`` using transformer-style naming."""

        return self.decode(
            graphs,
            target=target,
            by=by,
            recursive=recursive,
            boundary_policy=boundary_policy,
            validate=validate,
        )

    def fit_artifacts(self, graphs: GraphInput) -> tuple[TreeEncoder, TreeDecoder]:
        """Fit and return ``(encoder, decoder)`` directly."""

        self.fit(graphs)
        assert self.encoder_ is not None and self.decoder_ is not None
        return self.encoder_, self.decoder_

    @abstractmethod
    def _fit(self, graphs: Sequence[nx.DiGraph]) -> tuple[TreeEncoder, TreeDecoder]:
        """Subclass implementation for fitting normalized trees."""
