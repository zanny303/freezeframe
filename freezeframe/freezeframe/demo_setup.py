"""Wires a ready-to-serve enclave from trained artifacts: engine (hot tier) + warm tier, primed
with the history just before the test split, plus the bank-side test payments to replay."""
import hashlib
import pickle

import pandas as pd

from . import config
from .consortium import assign_members
from .enclave.crypto import KeyPair
from .enclave.engine import Engine
from .enclave.warm import WarmProcess, WarmTier
from .experiment import enclave_key
from .ingest import load, rates
from .pseudonym import Pseudonymizer

# GFP does not strictly bound look-back by its windows (tests/test_features.py measures the
# drift with a 48 h prime), so the enclave primes from all prior history for exact features.
PRIME_SECONDS = None


def _bundle_path(dataset: str, tier: str):
    return config.ARTIFACTS_DIR / "models" / dataset / f"{tier}.pkl"


def bundle_file_digest(dataset: str, tier: str) -> str:
    return hashlib.sha256(_bundle_path(dataset, tier).read_bytes()).hexdigest()


def load_bundle(dataset: str, tier: str) -> dict:
    """Load a model bundle; ``digest`` (SHA-256 of the file) goes into the attestation document.
    Bundles are pickles: only load ones this consortium trained (they are code)."""
    data = _bundle_path(dataset, tier).read_bytes()
    bundle = pickle.loads(data)
    bundle["digest"] = hashlib.sha256(data).hexdigest()
    return bundle


def to_payment(row) -> dict:
    return {"ref": str(row.txn_id), "ts": str(row.ts), "from_bank": str(row.from_bank),
            "from_acct": row.from_acct, "to_bank": str(row.to_bank), "to_acct": row.to_acct,
            "amount_paid": float(row.amt_paid_f), "cur_paid": str(row.cur_paid),
            "cur_recv": str(row.cur_recv), "fmt": str(row.fmt)}


def build_engine(dataset: str = "HI-Small", seed: int = 0, threads: int = 8, warm_interval=3.0,
                 prime: bool = True, live_from_t: int | None = None, prime_warm: bool = True,
                 warm_process: bool = False):
    """Engine + warm tier primed with the PRIME_SECONDS of history before ``live_from_t``
    (default: the start of the test split). ``ctx["live"]`` holds the bank-side payments to replay."""
    raw = load(dataset)
    bank_member = assign_members(raw, seed)
    key = enclave_key()
    hot, full = load_bundle(dataset, "hot"), load_bundle(dataset, "full")
    engine = Engine(hot, bank_member, key, rates(dataset), t0=raw.ts.min(), threads=threads)
    test = raw[raw.split == 2]
    t_live = int(test.t.iloc[0]) if live_from_t is None else int(live_from_t)
    if warm_process:   # separate interpreter; primes itself from the same history
        warm = WarmProcess(engine, dataset, t_live, interval_s=warm_interval,
                           threads=max(1, threads // 2))
        prime_warm = False
    else:
        warm = WarmTier(engine, full, interval_s=warm_interval, threads=max(1, threads // 2))
    if prime:
        lo = -1 if PRIME_SECONDS is None else t_live - PRIME_SECONDS
        hist = raw[(raw.t < t_live) & (raw.t >= lo)]
        hist = Pseudonymizer(key).apply(hist)
        engine.prime(hist)
        if prime_warm:
            warm.prime(hist)
    ctx = {"raw": raw, "test": test, "live": raw[raw.t >= t_live], "bank_member": bank_member}
    return engine, KeyPair(), warm, ctx


def chain_candidates(raw: pd.DataFrame, bank_member: dict, min_members: int = 3,
                     max_span_hours: float = 6) -> list:
    """Real laundering attempts in the test split that cross >= min_members members and pass
    money on (some receiving account later sends), short enough to replay in a demo."""
    attempts = raw[raw.attempt >= 0]
    in_test = attempts.groupby("attempt").split.min() == 2   # every hop is unseen test data
    out = []
    for aid, g in attempts[attempts.attempt.isin(in_test[in_test].index)].groupby("attempt"):
        g = g.sort_values("ts")
        members = set(g.from_bank.astype(str).map(bank_member)) | set(
            g.to_bank.astype(str).map(bank_member))
        recv = set(zip(g.to_bank.astype(str), g.to_acct))
        onward = any((b, a) in recv for b, a in zip(g.from_bank.astype(str), g.from_acct))
        span = (g.ts.iloc[-1] - g.ts.iloc[0]).total_seconds() / 3600
        if len(members) >= min_members and onward and 3 <= len(g) <= 12 and span <= max_span_hours:
            out.append({"attempt": int(aid), "pattern": str(g.pattern.iloc[0]), "hops": len(g),
                        "members": len(members), "span_hours": span})
    return sorted(out, key=lambda c: (-c["members"], c["span_hours"]))
