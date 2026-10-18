"""Consortium membership: the dataset's banks are dealt at random to N members.

A member's *view* is every transaction its banks send or receive (what a real bank sees).
A transaction is *owned* by the member of its sending bank; scores and decisions go only there.
"""
import numpy as np
import pandas as pd

from .config import N_MEMBERS


def assign_members(df: pd.DataFrame, seed: int, n_members: int = N_MEMBERS) -> dict:
    banks = np.unique(np.concatenate([df.from_bank.astype(str).unique(),
                                      df.to_bank.astype(str).unique()]))
    rng = np.random.default_rng(seed)
    members = rng.integers(0, n_members, size=len(banks))
    return dict(zip(banks, members.tolist()))


def owner(df: pd.DataFrame, bank_member: dict) -> np.ndarray:
    return df.from_bank.astype(str).map(bank_member).to_numpy(np.int8)


def receiver(df: pd.DataFrame, bank_member: dict) -> np.ndarray:
    return df.to_bank.astype(str).map(bank_member).to_numpy(np.int8)


def view_mask(df: pd.DataFrame, bank_member: dict, member: int) -> np.ndarray:
    return (owner(df, bank_member) == member) | (receiver(df, bank_member) == member)
