"""Evaluation: minority-class F1, recall under a review budget, laundered dollars stopped."""
import numpy as np
import pandas as pd

from .config import REVIEW_FRACTION
from .decide import FREEZE, HOLD, RELEASE


def f1_at(p, y, threshold) -> dict:
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def recall_at_fraction(p, y, fraction=REVIEW_FRACTION) -> float:
    k = max(1, int(round(len(p) * fraction)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate(p, y, amount_usd, decisions, threshold, patterns=None) -> dict:
    y = np.asarray(y).astype(np.int8)
    laundered = y == 1
    stopped = laundered & (decisions != RELEASE)
    out = {
        "n": int(len(y)),
        "positives": int(laundered.sum()),
        **f1_at(p, y, threshold),
        "recall_at_review_budget": recall_at_fraction(p, y),
        "laundered_usd": float(amount_usd[laundered].sum()),
        "laundered_usd_stopped": float(amount_usd[stopped].sum()),
        "holds": int((decisions == HOLD).sum()),
        "freezes": int((decisions == FREEZE).sum()),
        "freeze_precision": float(y[decisions == FREEZE].mean()) if (decisions == FREEZE).any() else None,
        "legit_payments_stopped": int((~laundered & (decisions != RELEASE)).sum()),
    }
    if patterns is not None:
        pat = pd.Series(np.asarray(patterns)).astype(str)
        hit = p >= threshold
        per = {}
        for name in sorted(set(pat[laundered]) - {""}):
            m = laundered & (pat.to_numpy() == name)
            per[name] = {"positives": int(m.sum()), "recall": float(hit[m].mean())}
        m = laundered & (pat.to_numpy() == "")
        if m.any():
            per["(unpatterned)"] = {"positives": int(m.sum()), "recall": float(hit[m].mean())}
        out["per_pattern_recall"] = per
    return out
