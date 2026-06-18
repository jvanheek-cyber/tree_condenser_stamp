"""Exception types used by tree_coarsening."""

from __future__ import annotations


class TreeCoarseningError(Exception):
    """Base exception for package-specific errors."""


class ValidationError(TreeCoarseningError, ValueError):
    """Raised when a graph, vocabulary entry, or rule violates the API contract."""


class NotFittedError(TreeCoarseningError, RuntimeError):
    """Raised when transform/decode is called before fitting."""
