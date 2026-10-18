"""Calibration and the release / hold / freeze rule. Plain code on calibrated probabilities."""
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import FREEZE_PRECISION, REVIEW_COST_USD

RELEASE, HOLD, FREEZE = 0, 1, 2
ACTIONS = ("release", "hold", "freeze")


@dataclass
class Policy:
    calibrator: IsotonicRegression
    f1_threshold: float       # operating point for the F1 metric
    freeze_threshold: float   # inf when no threshold reaches the target precision on tune
    review_cost_usd: float = REVIEW_COST_USD

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(raw)

    def decide(self, p: np.ndarray, amount_usd: np.ndarray) -> np.ndarray:
        """p is calibrated. Freeze outranks hold; hold when expected loss exceeds review cost."""
        out = np.full(len(p), RELEASE, np.int8)
        out[p * amount_usd > self.review_cost_usd] = HOLD
        out[p >= self.freeze_threshold] = FREEZE
        return out


def best_f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    # only cut where the score changes, so the threshold is realisable
    last_of_run = np.append(ps[1:] != ps[:-1], True)
    f1 = np.where(last_of_run, f1, -1)
    i = int(np.argmax(f1))
    return float(ps[i])


def precision_threshold(p: np.ndarray, y: np.ndarray, target: float = FREEZE_PRECISION,
                        min_tp: int = 5) -> float:
    """Lowest threshold whose precision on (p, y) is >= target, with at least min_tp hits."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    last_of_run = np.append(ps[1:] != ps[:-1], True)
    ok = np.flatnonzero(last_of_run & (prec >= target) & (tp >= min_tp))
    return float(ps[ok[-1]]) if len(ok) else float("inf")


def fit_policy(raw_tune: np.ndarray, y_tune: np.ndarray) -> Policy:
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(raw_tune, y_tune)
    p = cal.predict(raw_tune)
    return Policy(cal, best_f1_threshold(p, y_tune), precision_threshold(p, y_tune))
