import numpy as np
from snapml import GraphFeaturePreprocessor

from freezeframe.config import GFP_FULL, GFP_HOT
from freezeframe.features import GraphFeatures, edge_matrix, find_hubs, raw_features
from freezeframe.pseudonym import SPLIT_VERTEX_BASE


def _edges(pseudo_df, hubs=None):
    return edge_matrix(pseudo_df, np.array([], np.int64) if hubs is None else hubs)


def _bounds(t):
    return np.append(np.flatnonzero(np.diff(t)) + 1, len(t))


def test_streaming_equals_batch(pseudo_df):
    """Micro-batches (any cut at timestamp boundaries) give the training features exactly."""
    E = _edges(pseudo_df)
    batch = GraphFeatures(GFP_FULL).stream(E)
    g = GraphFeatures(GFP_FULL)
    b = _bounds(E[:, 3])
    starts = np.concatenate([[0], b[:-1]])
    # irregular micro-batches: 1..7 timestamps each
    rng = np.random.default_rng(1)
    parts, i = [], 0
    while i < len(starts):
        j = min(len(starts), i + int(rng.integers(1, 8)))
        seg = E[starts[i]:b[j - 1]]
        t = seg[:, 3]
        for s, e in zip(np.concatenate([[0], _bounds(t)[:-1]]), _bounds(t)):
            parts.append(g.update(seg[s:e]))
        i = j
    np.testing.assert_allclose(np.vstack(parts), batch, rtol=1e-6)


def test_no_lookahead(pseudo_df):
    """A payment's features do not change when later payments arrive."""
    E = _edges(pseudo_df)
    full = GraphFeatures(GFP_FULL).stream(E)
    cut = _bounds(E[:, 3])[len(_bounds(E[:, 3])) // 2]
    prefix = GraphFeatures(GFP_FULL).stream(E[:cut])
    np.testing.assert_allclose(prefix, full[:cut], rtol=1e-6)


def test_fit_transform_double_counts(pseudo_df):
    """Documents the GFP behaviour features.py avoids: fit_transform inserts edges twice."""
    E = _edges(pseudo_df)[:500]
    params = {**GFP_HOT, "num_threads": 1, "vertex_stats": False}
    g = GraphFeaturePreprocessor()
    g.set_params(params)
    ft = g.fit_transform(E)[:, 5:]
    g2 = GraphFeaturePreprocessor()
    g2.set_params(params)
    tr = g2.transform(E)[:, 5:]
    assert not np.allclose(ft, tr)


def test_hub_splitting(pseudo_df):
    split = pseudo_df.split.to_numpy()
    hubs = find_hubs(pseudo_df.src.to_numpy(), pseudo_df.dst.to_numpy(), split == 0, threshold=300)
    assert len(hubs) >= 1
    E = edge_matrix(pseudo_df, hubs)
    hub_rows = np.isin(pseudo_df.dst.to_numpy(), hubs)
    assert (E[hub_rows, 2] >= SPLIT_VERTEX_BASE).all()
    assert len(np.unique(E[hub_rows, 2])) == hub_rows.sum()   # one fresh vertex per edge
    assert (E[:, 1:3] < 2 ** 53).all()                         # exact in float64
    X = raw_features(pseudo_df, hubs)
    assert X[hub_rows, -1].all()


def test_priming(pseudo_df):
    """Priming from all prior history reproduces training features exactly. A 48 h prime
    (twice the longest window) drifts on a few rows: GFP's windows are not hard look-back
    bounds, which is why demo_setup primes from full history."""
    split = pseudo_df.split.to_numpy()
    hubs = find_hubs(pseudo_df.src.to_numpy(), pseudo_df.dst.to_numpy(), split == 0, threshold=300)
    E = edge_matrix(pseudo_df, hubs)
    full = GraphFeatures(GFP_FULL).stream(E)
    t = E[:, 3]
    live = np.searchsorted(t, t[int(len(t) * 0.8)])
    g = GraphFeatures(GFP_FULL)
    g.stream(E[:live])
    np.testing.assert_allclose(g.stream(E[live:]), full[live:], rtol=1e-6)
    g48 = GraphFeatures(GFP_FULL)
    g48.stream(E[np.searchsorted(t, t[live] - 48 * 3600):live])
    drift = ~np.isclose(g48.stream(E[live:]), full[live:], rtol=1e-5)
    assert drift.any(1).mean() < 0.02


def test_feature_groups_match_gfp_layout(pseudo_df):
    """Reason codes rely on GFP emitting each pattern as one contiguous block, in GFP_ORDER.
    Check it: each pattern computed alone must equal its block of the combined output."""
    from freezeframe.features import GFP_ORDER, RAW_FEATURES, RAW_GROUPS, feature_groups
    E = _edges(pseudo_df)[:3000]
    for params in (GFP_HOT, GFP_FULL):
        g = GraphFeatures(params, threads=1)   # single-threaded: GFP threading is flaky 
        full = g.stream(E)
        start = 0
        for name, width in g.layout:
            only = {**params, **{k: False for k in GFP_ORDER if k != name}}
            np.testing.assert_allclose(full[:, start:start + width],
                                       GraphFeatures(only, threads=1).stream(E), rtol=1e-6)
            start += width
        assert start == g.n_features
        assert len(feature_groups(g)) == len(RAW_FEATURES) + g.n_features == len(RAW_GROUPS) + g.n_features


def test_live_raw_features_match_batch(pseudo_df):
    """The enclave's list-based fast path must produce exactly the training columns."""
    from freezeframe.features import raw_features_live
    d = pseudo_df.head(500)
    hubs = find_hubs(pseudo_df.src.to_numpy(), pseudo_df.dst.to_numpy(),
                     pseudo_df.split.to_numpy() == 0, threshold=300)
    ts = d.ts.dt.to_pydatetime()
    live = raw_features_live(d.amount_usd.to_numpy(), d.fmt.astype(str).tolist(),
                             d.cur_paid.astype(str).tolist(), d.cur_recv.astype(str).tolist(),
                             d.from_bank.astype(str).tolist(), d.to_bank.astype(str).tolist(),
                             d.src.to_numpy(), d.dst.to_numpy(), [x.hour for x in ts],
                             [x.weekday() for x in ts], hubs)
    np.testing.assert_array_equal(live, raw_features(d, hubs))
