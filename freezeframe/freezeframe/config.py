"""Paths and tunables shared across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

# Time-ordered split: train / tune (calibration + thresholds) / test.
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

N_MEMBERS = 5
SEEDS = (0, 1, 2)

# Accounts whose train-split degree exceeds this are "hubs". AMLworld has two accounts with
# >100k edges; their adjacency lists make fan/degree/scatter-gather mining quadratic. Each hub
# endpoint is replaced by a fresh vertex per edge ("hub splitting") so the counterparty's
# patterns are still mined, and the model gets explicit is-hub flags instead.
HUB_DEGREE = 10_000

# Review budget for recall@k: the share of a bank's transactions an analyst team can review.
REVIEW_FRACTION = 0.005
# Expected laundered dollars (p * amount) above which a payment is held for review.
REVIEW_COST_USD = 100.0
# Freeze threshold is the lowest calibrated probability reaching this precision on the tune split.
FREEZE_PRECISION = 0.95

HOUR = 3600
_BINS = [2, 3, 4, 5, 10, 20]
_CYCLE_BINS = [2, 3, 4, 5, 10]

# Hot tier: every payment, inside the latency budget.
GFP_HOT = {
    "time_window": 24 * HOUR,
    "vertex_stats": True,
    "vertex_stats_cols": [4],  # amount_usd
    "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10], "vertex_stats_tw": 24 * HOUR,
    "fan": True, "fan_tw": 24 * HOUR, "fan_bins": _BINS,
    "degree": True, "degree_tw": 24 * HOUR, "degree_bins": _BINS,
    "scatter-gather": False,
    "temp-cycle": True, "temp-cycle_tw": 24 * HOUR, "temp-cycle_bins": _CYCLE_BINS,
    "lc-cycle": True, "lc-cycle_tw": 24 * HOUR, "lc-cycle_len": 8, "lc-cycle_bins": _CYCLE_BINS,
}

# Full / warm tier: adds scatter-gather, the most expensive pattern.
GFP_FULL = {**GFP_HOT, "scatter-gather": True, "scatter-gather_tw": 24 * HOUR,
            "scatter-gather_bins": _BINS}

GFP_THREADS = 8
