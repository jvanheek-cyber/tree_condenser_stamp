from __future__ import annotations

from tree_coarsening import StarCoarsener, combine
from tree_coarsening.utils import make_starburst_dataset


def test_imports_and_lazy_combine() -> None:
    X = make_starburst_dataset(n_graphs=2, seed=12, max_nodes=10, n_bursts=2, burst_size_range=(4, 4))
    c = StarCoarsener(d=3, m=1).fit(X)
    enc, dec = combine([c.encoder_], [c.decoder_])
    H = enc.encode(X[0])
    G = dec.decode(H)
    assert G.number_of_nodes() == X[0].number_of_nodes()
