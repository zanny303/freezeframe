import numpy as np
import pandas as pd
import pytest

from freezeframe import config
from freezeframe.decide import fit_policy
from freezeframe.features import build_features, find_hubs
from freezeframe.model import train
from freezeframe.pseudonym import Pseudonymizer

KEY = b"test-key-0123456789abcdef-012345"


def synthetic(n=6000, n_banks=12, n_accts=400, seed=0) -> pd.DataFrame:
    """A small AMLworld-shaped frame (raw account strings, time-sorted, with splits)."""
    rng = np.random.default_rng(seed)
    banks = [f"{i:03d}" for i in range(n_banks)]
    acct_bank = rng.integers(0, n_banks, n_accts)
    src = rng.integers(0, n_accts, n)
    dst = rng.integers(0, n_accts, n)
    dst[: n // 10] = 7                      # one busy account, to exercise hub splitting
    t = np.sort(rng.integers(0, 3 * 86400, n) // 60 * 60)
    y = (rng.random(n) < 0.03).astype(np.int8)
    amt = np.round(rng.lognormal(7, 1.5, n), 2) * (1 + 4 * y)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(t, unit="s")
    df = pd.DataFrame({
        "ts": ts, "t": t,
        "from_bank": [banks[acct_bank[a]] for a in src], "from_acct": [f"A{a:05X}" for a in src],
        "to_bank": [banks[acct_bank[a]] for a in dst], "to_acct": [f"A{a:05X}" for a in dst],
        "cur_recv": "US Dollar", "cur_paid": "US Dollar",
        "fmt": rng.choice(["ACH", "Cheque", "Wire", "Cash"], n),
        "y": y, "amount_usd": amt, "amt_paid_f": amt,
        "pattern": np.where(y == 1, "FAN-OUT", ""), "attempt": np.where(y == 1, 0, -1),
    })
    df["txn_id"] = np.arange(n)
    split = np.zeros(n, np.int8)
    split[int(n * 0.6):int(n * 0.8)] = 1
    split[int(n * 0.8):] = 2
    df["split"] = split
    return df


@pytest.fixture(scope="session")
def raw_df():
    return synthetic()


@pytest.fixture(scope="session")
def pseudo_df(raw_df):
    return Pseudonymizer(KEY).apply(raw_df)


@pytest.fixture(scope="session")
def bundles(pseudo_df):
    """Tiny trained hot/full bundles in the same format experiment.py writes."""
    out = {}
    split = pseudo_df.split.to_numpy()
    y = pseudo_df.y.to_numpy()
    hubs = find_hubs(pseudo_df.src.to_numpy(), pseudo_df.dst.to_numpy(), split == 0, threshold=300)
    for tier, params in (("hot", config.GFP_HOT), ("full", config.GFP_FULL)):
        X = build_features(pseudo_df, hubs, params)
        s = train(X[split == 0], y[split == 0], X[split == 1], y[split == 1], kind="xgb",
                  params={"n_estimators": 30, "early_stopping_rounds": 10}, n_jobs=2)
        raw = s.predict(X[split == 1])
        out[tier] = {"tier": tier, "params": params, "scorer": s,
                     "policy": fit_policy(raw, y[split == 1]), "hubs": hubs,
                     "n_features": X.shape[1]}
    return out
