"""Non-core utilities for simulation and bug-testing."""

from .simulate import (
    add_starbursts,
    make_edge_bpe_dataset,
    make_named_component_tree,
    make_repeated_edge_tree,
    make_starburst_dataset,
    random_galton_watson_tree,
)

__all__ = [
    "add_starbursts",
    "make_edge_bpe_dataset",
    "make_named_component_tree",
    "make_repeated_edge_tree",
    "make_starburst_dataset",
    "random_galton_watson_tree",
]
