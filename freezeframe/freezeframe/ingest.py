"""AMLworld CSV -> typed, time-sorted parquet with USD amounts, split labels and pattern labels.

This runs on the bank side of the boundary: raw account strings still exist here and are
only turned into pseudonyms when transactions enter the enclave (see pseudonym.py).
"""
import argparse
import re

import numpy as np
import pandas as pd

from .config import DATA_DIR, PROCESSED_DIR, SPLIT_FRACTIONS

COLUMNS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv",
           "amt_paid", "cur_paid", "fmt", "y"]
_STR = {c: str for c in COLUMNS}
SPLITS = ("train", "tune", "test")


def read_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=_STR, header=0, names=COLUMNS)
    return df


def usd_rates(df: pd.DataFrame) -> dict:
    """Per-currency USD rate, read off the dataset's own cross-currency payments."""
    rates = {"US Dollar": 1.0}
    paid = df.amt_paid.astype(float)
    recv = df.amt_recv.astype(float)
    ok = (paid > 0) & (recv > 0)
    to_usd = ok & (df.cur_recv == "US Dollar") & (df.cur_paid != "US Dollar")
    for cur, r in (recv[to_usd] / paid[to_usd]).groupby(df.cur_paid[to_usd]).median().items():
        rates[cur] = float(r)
    from_usd = ok & (df.cur_paid == "US Dollar") & (df.cur_recv != "US Dollar")
    for cur, r in (paid[from_usd] / recv[from_usd]).groupby(df.cur_recv[from_usd]).median().items():
        rates.setdefault(cur, float(r))
    missing = set(df.cur_paid.unique()) - set(rates)
    if missing:
        raise ValueError(f"no USD rate derivable for {sorted(missing)}")
    return rates


def parse_patterns(path) -> pd.DataFrame:
    """Rows of *_Patterns.txt, each tagged with its attempt id and pattern type."""
    rows, attempt, kind = [], -1, None
    head = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = head.match(line)
            if m:
                attempt += 1
                kind = m.group(1)
            elif line.startswith("END"):
                kind = None
            elif kind:
                rows.append(line.split(",")[:10] + [attempt, kind])
    return pd.DataFrame(rows, columns=COLUMNS[:10] + ["attempt", "pattern"])


def build(name: str) -> pd.DataFrame:
    df = read_raw(DATA_DIR / f"{name}_Trans.csv")
    df["ts"] = pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M")
    rates = usd_rates(df)
    df["amount_usd"] = df.amt_paid.astype(float) * df.cur_paid.map(rates)
    df["amt_paid_f"] = df.amt_paid.astype(float)
    df["y"] = df.y.astype(np.int8)

    pat = parse_patterns(DATA_DIR / f"{name}_Patterns.txt")
    pat["ts"] = pd.to_datetime(pat.ts, format="%Y/%m/%d %H:%M")
    key = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_paid"]
    pat = pat.drop_duplicates(key)[key + ["attempt", "pattern"]]
    df = df.merge(pat, on=key, how="left")
    df["attempt"] = df.attempt.fillna(-1).astype(np.int32)
    df["pattern"] = df.pattern.fillna("")

    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df), dtype=np.int64)
    df["t"] = (df.ts - df.ts.min()).dt.total_seconds().astype(np.int64)

    n = len(df)
    cut1 = int(n * SPLIT_FRACTIONS[0])
    cut2 = int(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    split = np.zeros(n, dtype=np.int8)
    split[cut1:cut2] = 1
    split[cut2:] = 2
    df["split"] = split

    for c in ["from_bank", "to_bank", "cur_recv", "cur_paid", "fmt", "pattern"]:
        df[c] = df[c].astype("category")
    return df.drop(columns=["amt_recv", "amt_paid"])


def rates(name: str) -> dict:
    """USD rates for a dataset, cached next to the parquet (the enclave needs them to score)."""
    import json
    path = PROCESSED_DIR / f"{name}_usd_rates.json"
    if not path.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(usd_rates(read_raw(DATA_DIR / f"{name}_Trans.csv")), indent=1))
    return json.loads(path.read_text())


def load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        build(name).to_parquet(path)
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="HI-Small")
    args = ap.parse_args()
    df = load(args.dataset)
    print(f"{args.dataset}: {len(df):,} transactions, {df.y.sum():,} laundering")
    for i, s in enumerate(SPLITS):
        part = df[df.split == i]
        print(f"  {s:5s} {len(part):>9,} rows  {part.ts.min()} .. {part.ts.max()}  "
              f"laundering 1 in {len(part) / max(part.y.sum(), 1):,.0f}")
    print(f"  pattern-labelled laundering: {(df.attempt >= 0).sum():,} of {df.y.sum():,}")


if __name__ == "__main__":
    main()
