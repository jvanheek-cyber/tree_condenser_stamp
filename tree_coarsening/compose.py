"""Composition helpers for fitted encoder/decoder artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from .decoder import LazyTreeDecoder, TreeDecoder
from .encoder import LazyTreeEncoder, TreeEncoder
from .exceptions import ValidationError
from .vocabulary import Token, TokenSpec, VocabEntry, Vocabulary


def combine(
    encoders: Sequence[TreeEncoder],
    decoders: Sequence[TreeDecoder],
    *,
    mode: Literal["lazy", "materialized"] = "lazy",
    validate: bool = True,
) -> tuple[TreeEncoder, TreeDecoder]:
    """Return a combined encoder/decoder pair.

    Encoders and decoders are supplied in encoder application order. Lazy
    composition applies encoders in the supplied order and decoders in reverse
    order. Materialized recipe substitution is reserved for a later pass.
    """

    encoders = tuple(encoders)
    decoders = tuple(decoders)
    if mode != "lazy":
        raise NotImplementedError("materialized composition is not implemented yet.")
    if len(encoders) != len(decoders):
        raise ValidationError("encoders and decoders must have the same length.")
    if not encoders:
        raise ValidationError("at least one encoder/decoder pair is required.")
    if validate:
        for i, (encoder, decoder) in enumerate(zip(encoders, decoders, strict=True)):
            if encoder.model_id != decoder.model_id:
                raise ValidationError(
                    f"encoder/decoder pair {i} has mismatched model ids: "
                    f"{encoder.model_id!r} vs {decoder.model_id!r}."
                )
            for attr_name in (
                "label_attr",
                "type_attr",
                "size_attr",
                "time_attr",
                "uid_attr",
                "super_label_attr",
                "super_uid_attr",
                "attach_attr",
            ):
                if getattr(encoder, attr_name) != getattr(decoder, attr_name):
                    raise ValidationError(
                        f"encoder/decoder pair {i} uses inconsistent {attr_name}."
                    )
            _validate_pair_vocab(encoder.vocab, decoder.vocab)

    model_id = f"combined:{uuid4().hex[:12]}"
    vocab = _merge_vocabularies([encoder.vocab for encoder in encoders])
    base_labels = frozenset().union(*(encoder.base_labels for encoder in encoders))

    combined_encoder = LazyTreeEncoder(
        model_id=model_id,
        vocab=vocab,
        rules=tuple(rule for encoder in encoders for rule in encoder.rules),
        base_labels=base_labels,
        label_attr=encoders[0].label_attr,
        type_attr=encoders[0].type_attr,
        size_attr=encoders[0].size_attr,
        time_attr=encoders[0].time_attr,
        uid_attr=encoders[0].uid_attr,
        super_label_attr=encoders[0].super_label_attr,
        super_uid_attr=encoders[0].super_uid_attr,
        attach_attr=encoders[0].attach_attr,
        encoders=encoders,
    )
    combined_decoder = LazyTreeDecoder(
        model_id=model_id,
        vocab=vocab,
        base_labels=base_labels,
        label_attr=decoders[0].label_attr,
        type_attr=decoders[0].type_attr,
        size_attr=decoders[0].size_attr,
        time_attr=decoders[0].time_attr,
        uid_attr=decoders[0].uid_attr,
        super_label_attr=decoders[0].super_label_attr,
        super_uid_attr=decoders[0].super_uid_attr,
        attach_attr=decoders[0].attach_attr,
        decoders=decoders,
    )
    return combined_encoder, combined_decoder


def _validate_pair_vocab(encoder_vocab: Vocabulary, decoder_vocab: Vocabulary) -> None:
    for token in encoder_vocab.creation_order:
        entry = encoder_vocab.entries[token]
        other = decoder_vocab.entries.get(token)
        if other is None:
            raise ValidationError(f"decoder vocabulary is missing token {token!r}.")
        if other != entry:
            raise ValidationError(f"encoder/decoder vocabularies disagree on token {token!r}.")


def _merge_vocabularies(vocabs: Sequence[Vocabulary]) -> Vocabulary:
    entries: dict[Token, VocabEntry] = {}
    symbols: dict[Token, TokenSpec] = {}
    order: list[Token] = []
    for vocab in vocabs:
        for token, spec in vocab.symbols.items():
            previous = symbols.get(token)
            if previous is not None and previous != spec:
                raise ValidationError(
                    f"incompatible symbol specification for {token!r}: "
                    f"{previous!r} versus {spec!r}."
                )
            if token in entries:
                continue
            symbols[token] = spec
        for token in vocab.creation_order:
            entry = vocab.entries[token]
            symbols.pop(token, None)
            if token in entries:
                if entries[token] != entry:
                    raise ValidationError(f"incompatible duplicate token {token!r} during combine.")
                continue
            entries[token] = entry
            order.append(token)
    return Vocabulary(entries=entries, creation_order=order, symbols=symbols)
