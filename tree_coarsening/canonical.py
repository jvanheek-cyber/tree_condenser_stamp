"""Canonical root/site coordinate helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .vocabulary import Token, VocabEntry, Vocabulary, root_count, site_count


def external_root_positions(entry: VocabEntry) -> tuple[int, ...]:
    """Return recipe positions with ``P[i] == -1``."""

    return entry.root_positions


def attachment_slice(
    entry: VocabEntry,
    i: int,
    vocab: Mapping[Token, VocabEntry] | Vocabulary,
) -> tuple[int, ...]:
    """Convenience wrapper around ``VocabEntry.attachment_slice``."""

    return entry.attachment_slice(i, vocab)


__all__ = ["attachment_slice", "external_root_positions", "root_count", "site_count"]
