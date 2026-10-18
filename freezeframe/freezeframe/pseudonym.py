"""Keyed pseudonyms for accounts. Runs first inside the enclave; the key never leaves it.

A pseudonym is HMAC-SHA256(key, "bank:account") truncated to 52 bits, so it is exact as a
float64 (the Graph Feature Preprocessor takes vertex ids as float64).
"""
import hashlib
import hmac
import os

import numpy as np
import pandas as pd

PSEUDONYM_BITS = 52
# Hub splitting draws fresh vertex ids from above the pseudonym range (still < 2**53).
SPLIT_VERTEX_BASE = 1 << PSEUDONYM_BITS


def new_key() -> bytes:
    return os.urandom(32)


def account_key(bank, acct) -> str:
    return f"{bank}:{acct}"


class Pseudonymizer:
    """``cache=False`` for the live enclave path: a cache would keep raw account strings in
    memory as dictionary keys. HMAC-SHA256 is cheap enough to recompute per payment."""

    def __init__(self, key: bytes, cache: bool = True):
        if len(key) < 16:
            raise ValueError("pseudonym key too short")
        self._key = key
        self._cache: dict[str, int] | None = {} if cache else None

    def one(self, bank: str, acct: str) -> int:
        k = account_key(bank, acct)
        if self._cache is not None and (pid := self._cache.get(k)) is not None:
            return pid
        d = hmac.new(self._key, k.encode(), hashlib.sha256).digest()
        pid = int.from_bytes(d[:8], "big") >> (64 - PSEUDONYM_BITS)
        if self._cache is not None:
            self._cache[k] = pid
        return pid

    def column(self, banks: pd.Series, accts: pd.Series) -> np.ndarray:
        """Vectorised over unique accounts; returns int64 pseudonyms aligned with the input."""
        keys = banks.astype(str) + ":" + accts.astype(str)
        codes, uniques = pd.factorize(keys)
        pids = np.fromiter((self.one(*u.split(":", 1)) for u in uniques), dtype=np.int64,
                           count=len(uniques))
        return pids[codes]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace raw account columns with pseudonyms. Raw account strings are dropped."""
        out = df.drop(columns=["from_acct", "to_acct"])
        out["src"] = self.column(df.from_bank, df.from_acct)
        out["dst"] = self.column(df.to_bank, df.to_acct)
        return out
