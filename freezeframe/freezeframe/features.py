"""Model inputs: raw transaction features plus Graph Feature Preprocessor (GFP) pattern features.

GFP is fed edges in time order through ``transform``, which inserts each batch into its
in-memory graph and then computes that batch's features. Batch and streaming use the same
call, so a micro-batch in the enclave gets exactly the features the experiment trained on.

Two GFP behaviours this module works around (both checked in tests/test_features.py):
- ``fit_transform`` inserts every edge twice (fit, then transform), doubling all counts.
  Only ``transform`` is used.
- Inside one ``transform`` call every edge sees every other edge of the batch, including
  later ones. Batches are therefore cut at timestamp boundaries: a payment sees only edges
  at or before its own timestamp (same-minute payments see each other, as in a live
  micro-batch).
"""
import numpy as np
import pandas as pd
from snapml import GraphFeaturePreprocessor

from .config import GFP_THREADS, HUB_DEGREE
from .pseudonym import SPLIT_VERTEX_BASE

FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro",
              "Mexican Peso", "Ruble", "Rupee", "Saudi Riyal", "Shekel", "Swiss Franc",
              "UK Pound", "US Dollar", "Yen", "Yuan"]
F32_MAX = float(np.finfo(np.float32).max)
RAW_FEATURES = ["log_amount_usd", "fmt", "cur_paid", "cur_recv", "cross_currency",
                "same_bank", "self_loop", "hour", "weekday", "src_is_hub", "dst_is_hub"]
# Evidence family of each raw column, for reason codes (explain.py).
RAW_GROUPS = ["amount", "channel", "channel", "channel", "channel",
              "route", "route", "timing", "timing", "route", "route"]
# GFP emits its pattern blocks contiguously in this order (checked in tests/test_features.py).
GFP_ORDER = ("fan", "degree", "scatter-gather", "temp-cycle", "lc-cycle", "vertex_stats")
GFP_GROUPS = {"fan": "fan", "degree": "degree", "scatter-gather": "scatter_gather",
              "temp-cycle": "cycle", "lc-cycle": "cycle", "vertex_stats": "account_profile"}


def find_hubs(src: np.ndarray, dst: np.ndarray, fit_mask: np.ndarray,
              threshold: int = HUB_DEGREE) -> np.ndarray:
    """Accounts whose degree on the fitting rows exceeds the threshold (sorted pseudonyms)."""
    ends = np.concatenate([src[fit_mask], dst[fit_mask]])
    ids, counts = np.unique(ends, return_counts=True)
    return ids[counts > threshold]


def _codes(values: pd.Series, vocab: list) -> np.ndarray:
    return pd.Categorical(values.astype(str), categories=vocab).codes.astype(np.float32)


_FMT_CODE = {v: i for i, v in enumerate(FORMATS)}
_CUR_CODE = {v: i for i, v in enumerate(CURRENCIES)}


def raw_features_live(amount_usd, fmt, cur_paid, cur_recv, from_bank, to_bank, src, dst, hour,
                      weekday, hubs) -> np.ndarray:
    """Same columns as raw_features, from plain lists/arrays (the enclave's per-batch fast path)."""
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    cols = [
        np.log1p(np.asarray(amount_usd, np.float64)),
        [_FMT_CODE.get(v, -1) for v in fmt],
        [_CUR_CODE.get(v, -1) for v in cur_paid],
        [_CUR_CODE.get(v, -1) for v in cur_recv],
        [a != b for a, b in zip(cur_paid, cur_recv)],
        [a == b for a, b in zip(from_bank, to_bank)],
        src == dst,
        hour,
        weekday,
        np.isin(src, hubs),
        np.isin(dst, hubs),
    ]
    return np.column_stack(cols).astype(np.float32)


def raw_features(df: pd.DataFrame, hubs: np.ndarray) -> np.ndarray:
    src, dst = df.src.to_numpy(), df.dst.to_numpy()
    ts = pd.to_datetime(df.ts)
    cols = [
        np.log1p(df.amount_usd.to_numpy()),
        _codes(df.fmt, FORMATS),
        _codes(df.cur_paid, CURRENCIES),
        _codes(df.cur_recv, CURRENCIES),
        (df.cur_paid.astype(str).to_numpy() != df.cur_recv.astype(str).to_numpy()),
        (df.from_bank.astype(str).to_numpy() == df.to_bank.astype(str).to_numpy()),
        src == dst,
        ts.dt.hour.to_numpy(),
        ts.dt.weekday.to_numpy(),
        np.isin(src, hubs),
        np.isin(dst, hubs),
    ]
    return np.column_stack(cols).astype(np.float32)


def edge_matrix(df: pd.DataFrame, hubs: np.ndarray) -> np.ndarray:
    """[edge id, src, dst, t, amount_usd] in GFP's format, with hub endpoints split."""
    return edge_arrays(df.txn_id.to_numpy(), df.src.to_numpy(), df.dst.to_numpy(),
                       df.t.to_numpy(), df.amount_usd.to_numpy(), hubs)


def edge_arrays(txn, src, dst, t, amount_usd, hubs) -> np.ndarray:
    txn = np.asarray(txn, np.int64)
    src = np.array(src, np.int64)
    dst = np.array(dst, np.int64)
    hs, hd = np.isin(src, hubs), np.isin(dst, hubs)
    src[hs] = SPLIT_VERTEX_BASE + 2 * txn[hs]
    dst[hd] = SPLIT_VERTEX_BASE + 2 * txn[hd] + 1
    return np.column_stack([txn, src, dst, np.asarray(t, np.int64),
                            np.asarray(amount_usd, np.float64)]).astype(np.float64)


class GraphFeatures:
    """Stateful GFP over a stream of time-ordered edges."""

    N_RAW = 5

    def __init__(self, params: dict, threads: int = GFP_THREADS):
        self.params = {**params, "num_threads": threads}
        self.gfp = GraphFeaturePreprocessor()
        self.gfp.set_params(self.params)
        self.n_features = self._width(self.params)
        self.layout = self._layout()

    @classmethod
    def _width(cls, params: dict) -> int:
        probe = GraphFeaturePreprocessor()
        probe.set_params(params)
        return probe.fit_transform(np.zeros((1, cls.N_RAW))).shape[1] - cls.N_RAW

    def _layout(self) -> list[tuple[str, int]]:
        """(pattern, width) blocks in output order, probed one pattern at a time."""
        blocks = []
        for g in GFP_ORDER:
            if self.params.get(g):
                only = {**self.params, **{k: False for k in GFP_ORDER if k != g}, "num_threads": 1}
                blocks.append((g, self._width(only)))
        if sum(w for _, w in blocks) != self.n_features:
            raise RuntimeError("unexpected Graph Feature Preprocessor output layout")
        return blocks

    @property
    def groups(self) -> list[str]:
        return [GFP_GROUPS[g] for g, w in self.layout for _ in range(w)]

    def update(self, edges: np.ndarray) -> np.ndarray:
        """Insert a time-ordered batch of edges and return its graph features (float32)."""
        if len(edges) == 0:
            return np.zeros((0, self.n_features), np.float32)
        out = self.gfp.transform(np.ascontiguousarray(edges, dtype=np.float64))[:, self.N_RAW:]
        # Vertex-stat moments can exceed float32 range (e.g. skew of identical amounts); clip so
        # they stay finite instead of becoming inf, which XGBoost rejects. NaN is kept (missing).
        return np.clip(out, -F32_MAX, F32_MAX).astype(np.float32)

    def stream(self, edges: np.ndarray, progress=None) -> np.ndarray:
        """Feed time-sorted edges one timestamp at a time (no look-ahead)."""
        t = edges[:, 3]
        if len(t) and np.any(np.diff(t) < 0):
            raise ValueError("edges must be sorted by timestamp")
        bounds = np.append(np.flatnonzero(np.diff(t)) + 1, len(t))
        out = np.empty((len(edges), self.n_features), np.float32)
        start = 0
        for i, end in enumerate(bounds):
            out[start:end] = self.update(edges[start:end])
            start = end
            if progress and i % 2000 == 0:
                progress(end, len(edges))
        return out


def feature_groups(graph: GraphFeatures) -> list[str]:
    """Evidence family of every model input column (raw columns, then graph columns)."""
    return RAW_GROUPS + graph.groups


def build_features(df: pd.DataFrame, hubs: np.ndarray, params: dict, progress=None) -> np.ndarray:
    """Raw + graph features for a time-ordered frame of pseudonymised transactions."""
    graph = GraphFeatures(params).stream(edge_matrix(df, hubs), progress=progress)
    return np.hstack([raw_features(df, hubs), graph])
