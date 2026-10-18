"""A small synthetic AMLworld-shaped dataset for a quick end-to-end run (no Kaggle download).

It writes exactly what ``ingest.load`` would produce for a real dataset, so the experiment,
the benchmark, the HTTP enclave and the dashboard all run on it unchanged:

    python -m freezeframe.synth                           # data/processed/SYNTH.parquet (~2 s)
    python -m freezeframe.experiment SYNTH --seeds 0      # alone vs together (~3-6 min)
    python -m freezeframe.experiment SYNTH --tier hot --together-only
    FF_DATASET=SYNTH python -m demo.app                   # dashboard on :8000

or just ``./scripts/quick_demo.sh``.

Background payments follow daily cycles, repeat counterparties, a few high-volume merchant
accounts and several currencies and formats. Laundering attempts are planted the way AMLworld
describes them (cycle, fan-out, fan-in, scatter-gather, stack) through mule accounts spread over
many banks, so most attempts cross consortium members, which is what pooling is for.

**These are smoke-test numbers, not results.** Everything quoted in README.md comes from the
real HI-Small run in results/HI-Small/. The dashboard labels a synthetic run as such.
"""
import argparse
import json

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, SPLIT_FRACTIONS

NAME = "SYNTH"
START = pd.Timestamp("2022-09-01")
FORMATS = ["ACH", "Cheque", "Credit Card", "Cash", "Wire", "Reinvestment", "Bitcoin"]
FORMAT_P = [0.34, 0.20, 0.25, 0.08, 0.08, 0.04, 0.01]
RATES = {"US Dollar": 1.0, "Euro": 1.17, "UK Pound": 1.31, "Yen": 0.0091, "Rupee": 0.0136,
         "Canadian Dollar": 0.79, "Swiss Franc": 1.04, "Bitcoin": 19500.0}
CUR = list(RATES)
CUR_P = [0.80, 0.06, 0.04, 0.03, 0.03, 0.02, 0.015, 0.005]


def _accounts(rng, n_accts, n_banks):
    size = np.minimum(rng.zipf(1.6, n_banks), 40).astype(float)
    bank_of = rng.choice(n_banks, n_accts, p=size / size.sum())
    banks = np.array([str(b) for b in rng.choice(np.arange(10, 40000), n_banks, replace=False)])
    names = np.array([f"{rng.integers(0x80000000, 0x8FFFFFFFF):09X}" for _ in range(n_accts)])
    return banks[bank_of], names


def generate(rows: int = 200_000, days: int = 10, n_banks: int = 40, n_accts: int = 12_000,
             attempts: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acct_bank, acct_name = _accounts(rng, n_accts + 3000, n_banks)
    regular = np.arange(n_accts)
    mules = np.arange(n_accts, n_accts + 3000)           # used only by laundering (plus noise)

    # ---- background --------------------------------------------------------------------------
    activity = rng.lognormal(0, 1.0, n_accts)
    merchants = rng.choice(n_accts, 25, replace=False)
    pull = np.ones(n_accts)
    pull[merchants] = 60.0
    favourites = rng.choice(n_accts, (n_accts, 6), p=pull / pull.sum())
    src = rng.choice(n_accts, rows, p=activity / activity.sum())
    dst = np.where(rng.random(rows) < 0.7, favourites[src, rng.integers(0, 6, rows)],
                   rng.choice(n_accts, rows, p=pull / pull.sum()))
    day = rng.integers(0, days, rows)
    minute = np.clip(rng.normal(13 * 60, 4 * 60, rows), 0, 1439).astype(int)
    t = day * 86400 + minute * 60
    usd = np.round(rng.lognormal(5.5 + 0.6 * np.log1p(activity[src]), 1.3), 2)
    fmt = rng.choice(FORMATS, rows, p=FORMAT_P)
    reinvest = fmt == "Reinvestment"
    dst[reinvest] = src[reinvest]
    bg = pd.DataFrame({"t": t, "src": src, "dst": dst, "usd": usd, "fmt": fmt, "y": 0,
                       "attempt": -1, "pattern": ""})

    # ---- laundering --------------------------------------------------------------------------
    rows_l = []
    kinds = ["CYCLE", "FAN-OUT", "FAN-IN", "SCATTER-GATHER", "STACK"]
    for a in range(attempts):
        kind = kinds[a % len(kinds)]
        t0 = int(rng.integers(0, days * 86400 - 86400)) // 60 * 60
        amount = float(rng.lognormal(8.6, 0.9))
        fmt_l = rng.choice(["Wire", "ACH", "Cheque", "Bitcoin"], p=[0.45, 0.35, 0.1, 0.1])
        gap = lambda: int(rng.integers(10, 180)) * 60
        k = int(rng.integers(4, 9))
        ring = rng.choice(mules, k, replace=False)
        if rng.random() < 0.3:                            # some attempts start from a real customer
            ring[0] = rng.choice(regular)
        tt, edges = t0, []
        if kind in ("CYCLE", "STACK"):
            path = list(ring) + ([ring[0]] if kind == "CYCLE" else [])
            for s, d in zip(path[:-1], path[1:]):
                edges.append((tt, s, d, amount))
                amount *= rng.uniform(0.95, 0.99)
                tt += gap()
        elif kind == "FAN-OUT":
            for d in ring[1:]:
                edges.append((tt, ring[0], d, amount / (k - 1)))
                tt += gap() // 3
        elif kind == "FAN-IN":
            for s in ring[1:]:
                edges.append((tt, s, ring[0], amount / (k - 1)))
                tt += gap() // 3
        else:                                             # SCATTER-GATHER via k-2 intermediates
            sink = ring[-1]
            for mid in ring[1:-1]:
                edges.append((tt, ring[0], mid, amount / (k - 2)))
                tt += gap() // 4
            tt += gap()
            for mid in ring[1:-1]:
                edges.append((tt, mid, sink, amount / (k - 2) * 0.97))
                tt += gap() // 4
        for (te, s, d, amt) in edges:
            rows_l.append((te, s, d, round(amt, 2), fmt_l, 1, a, kind))
    lau = pd.DataFrame(rows_l, columns=["t", "src", "dst", "usd", "fmt", "y", "attempt", "pattern"])
    # mules also make a little ordinary-looking traffic, so "is a mule account" alone isn't a label
    n_noise = len(lau) * 2
    noise = pd.DataFrame({
        "t": rng.integers(0, days * 86400, n_noise) // 60 * 60,
        "src": rng.choice(mules, n_noise), "dst": rng.choice(n_accts, n_noise),
        "usd": np.round(rng.lognormal(5.5, 1.2, n_noise), 2),
        "fmt": rng.choice(FORMATS, n_noise, p=FORMAT_P), "y": 0, "attempt": -1, "pattern": ""})
    df = pd.concat([bg, lau, noise], ignore_index=True)

    # ---- AMLworld columns --------------------------------------------------------------------
    n = len(df)
    cur = rng.choice(CUR, n, p=CUR_P)
    cur[df.fmt.to_numpy() == "Bitcoin"] = "Bitcoin"
    cur_recv = np.where(rng.random(n) < 0.97, cur, rng.choice(CUR[:4], n))
    rate = pd.Series(cur).map(RATES).to_numpy()
    out = pd.DataFrame({
        "ts": START + pd.to_timedelta(df.t.to_numpy(), unit="s"),
        "from_bank": acct_bank[df.src], "from_acct": acct_name[df.src],
        "to_bank": acct_bank[df.dst], "to_acct": acct_name[df.dst],
        "cur_recv": cur_recv, "cur_paid": cur, "fmt": df.fmt.to_numpy(),
        "y": df.y.to_numpy().astype(np.int8),
        "amount_usd": df.usd.to_numpy(), "amt_paid_f": np.round(df.usd.to_numpy() / rate, 2),
        "attempt": df.attempt.to_numpy().astype(np.int32), "pattern": df.pattern.to_numpy(),
    })
    out["amount_usd"] = out.amt_paid_f * rate             # exactly what ingest would compute
    out = out.sort_values("ts", kind="stable").reset_index(drop=True)
    out["txn_id"] = np.arange(len(out), dtype=np.int64)
    out["t"] = (out.ts - out.ts.min()).dt.total_seconds().astype(np.int64)
    cut1 = int(len(out) * SPLIT_FRACTIONS[0])
    cut2 = int(len(out) * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    split = np.zeros(len(out), np.int8)
    split[cut1:cut2], split[cut2:] = 1, 2
    out["split"] = split
    for c in ["from_bank", "to_bank", "cur_recv", "cur_paid", "fmt", "pattern"]:
        out[c] = out[c].astype("category")
    return out


def write(name: str = NAME, **kw) -> pd.DataFrame:
    df = generate(**kw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet")
    (PROCESSED_DIR / f"{name}_usd_rates.json").write_text(json.dumps(RATES, indent=1))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--rows", type=int, default=200_000, help="background payments")
    ap.add_argument("--attempts", type=int, default=420, help="planted laundering attempts")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    df = write(a.name, rows=a.rows, attempts=a.attempts, seed=a.seed)
    print(f"{a.name}: {len(df):,} payments, {int(df.y.sum()):,} laundering in "
          f"{df[df.attempt >= 0].attempt.nunique()} attempts, "
          f"{pd.concat([df.from_bank, df.to_bank]).nunique()} banks -> {PROCESSED_DIR}/{a.name}.parquet")
    print("synthetic smoke-test data: numbers from it are not results")


if __name__ == "__main__":
    main()
