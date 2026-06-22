from __future__ import annotations

import networkx as nx
import pytest

from tree_coarsening import StagedTreeDecoder, VocabEntry, Vocabulary, base_token
from tree_coarsening.exceptions import ValidationError
from tree_coarsening.nx_io import edge_attach_attrs
from tree_coarsening.provenance import PROVENANCE_KEY, NODE_ATTRS_KEY
from tree_coarsening.validation import validate_encoded_tree


def test_flat_a_distinguishes_attachment_sites() -> None:
    vocab = Vocabulary()
    ab = ("edge", "AB")
    vocab.add(
        VocabEntry(
            token=ab,
            parent=(-1, 0),
            label=(base_token("A"), base_token("B")),
            attach=(0,),
            created_at_step=0,
            operation="edge",
        )
    )
    site0 = ("edge", "ABC", 0)
    site1 = ("edge", "ABC", 1)
    vocab.add(
        VocabEntry(
            token=site0,
            parent=(-1, 0),
            label=(ab, base_token("C")),
            attach=(0,),
            created_at_step=1,
            operation="edge",
        )
    )
    vocab.add(
        VocabEntry(
            token=site1,
            parent=(-1, 0),
            label=(ab, base_token("C")),
            attach=(1,),
            created_at_step=2,
            operation="edge",
        )
    )

    decoder = StagedTreeDecoder(model_id="m", vocab=vocab)
    for token, expected_edges in [
        (site0, {("a", "b"), ("a", "c")}),
        (site1, {("a", "b"), ("b", "c")}),
    ]:
        H = nx.DiGraph()
        H.graph[PROVENANCE_KEY] = {
            NODE_ATTRS_KEY: {
                "a": {"label": "A", "time": 0.0, "uid": "a"},
                "b": {"label": "B", "time": 1.0, "uid": "b"},
                "c": {"label": "C", "time": 2.0, "uid": "c"},
            }
        }
        H.add_node(0, label=token, super_uids=("a", "b", "c"))
        G = decoder.decode(H)
        assert set(G.edges) == expected_edges


def test_broad_sibling_boundary_policy() -> None:
    vocab = Vocabulary()
    ab = ("edge", "AB")
    cd = ("siblings", "CD")
    vocab.add(
        VocabEntry(
            token=ab,
            parent=(-1, 0),
            label=(base_token("A"), base_token("B")),
            attach=(0,),
            created_at_step=0,
            operation="edge",
        )
    )
    vocab.add(
        VocabEntry(
            token=cd,
            parent=(-1, -1),
            label=(base_token("C"), base_token("D")),
            attach=(),
            created_at_step=1,
            operation="siblings",
        )
    )
    H = nx.DiGraph()
    H.graph[PROVENANCE_KEY] = {
        NODE_ATTRS_KEY: {
            "a": {"label": "A", "time": 0.0, "uid": "a"},
            "b": {"label": "B", "time": 1.0, "uid": "b"},
            "c": {"label": "C", "time": 2.0, "uid": "c"},
            "d": {"label": "D", "time": 3.0, "uid": "d"},
        }
    }
    H.add_node(0, label=ab, super_uids=("a", "b"))
    H.add_node(1, label=cd, super_uids=("c", "d"))
    H.add_edge(0, 1, **edge_attach_attrs((0, 1)))
    validate_encoded_tree(H, vocab=vocab)

    decoder = StagedTreeDecoder(model_id="m", vocab=vocab)
    G = decoder.decode(H)
    assert set(G.edges) == {("a", "b"), ("a", "c"), ("b", "d")}

    with pytest.raises(ValidationError):
        decoder.decode(H, target=0, by="node", recursive=False, boundary_policy="raise")

    H2 = decoder.decode(H, target=0, by="node", recursive=False, boundary_policy="expand")
    assert all(data["type"] == data["label"] for _, data in H2.nodes(data=True))
    assert all(data["size"] == len(data["super_uids"]) for _, data in H2.nodes(data=True))
    assert set(decoder.decode(H2).edges) == {("a", "b"), ("a", "c"), ("b", "d")}


def test_root_and_site_counts_are_cached_for_deep_vocabularies() -> None:
    vocab = Vocabulary()
    previous = base_token("A")
    depth = 2500
    for i in range(depth):
        token = ("deep", i)
        vocab.add(
            VocabEntry(
                token=token,
                parent=(-1, 0),
                label=(previous, base_token("B")),
                attach=(0,),
                created_at_step=i,
                operation="edge",
            )
        )
        previous = token

    assert vocab.root_count(previous) == 1
    assert vocab.site_count(previous) == depth + 1
    # Repeated lookups should use the cached scalar rather than recurse through
    # the complete staged recipe chain.
    for _ in range(100):
        assert vocab.site_count(previous) == depth + 1
