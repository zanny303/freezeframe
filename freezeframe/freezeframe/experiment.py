"""The core experiment: laundering caught by each consortium member alone vs. inside the enclave.

Both arms score the same test transactions (each member's *owned* payments):
- alone:    GFP + booster trained, calibrated and thresholded on that member's view only.
- together: one pipeline over every member's pseudonymised transactions; each score goes back
            to the member that owns the payment.

Run:  arch -x86_64 .venv/bin/python -m freezeframe.experiment HI-Small
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .consortium import assign_members, owner, view_mask
from .decide import FREEZE, RELEASE, fit_policy
from .features import build_features, find_hubs
from .ingest import load
from .metrics import evaluate, f1_at
from .model import train
from .pseudonym import Pseudonymizer, new_key

TIERS = {"hot": config.GFP_HOT, "full": config.GFP_FULL}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def enclave_key() -> bytes:
    """Simulated enclave sealed storage: one persistent key under artifacts/ (gitignored)."""
    path = config.ARTIFACTS_DIR / "enclave_pseudonym.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:   # created 0600 in one step: never world-readable, even briefly
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:   # another process created it first
            return path.read_bytes()
        with os.fdopen(fd, "wb") as f:
            f.write(new_key())
    return path.read_bytes()


def pseudonymised(dataset: str) -> pd.DataFrame:
    return Pseudonymizer(enclave_key()).apply(load(dataset))


def run_pipeline(df: pd.DataFrame, params: dict, model_kind: str, X: np.ndarray | None = None):
    """Features -> train on split 0 -> calibrate/threshold on split 1. Returns (X, scorer, hubs)."""
    split = df.split.to_numpy()
    hubs = find_hubs(df.src.to_numpy(), df.dst.to_numpy(), split == 0)
    if X is None:
        X = build_features(df, hubs, params)
    y = df.y.to_numpy()
    tr, tu = split == 0, split == 1
    scorer = train(X[tr], y[tr], X[tu], y[tu], kind=model_kind)
    return X, scorer, hubs


def score_and_policy(df, X, scorer, policy_mask):
    raw = scorer.predict(X)
    y = df.y.to_numpy()
    policy = fit_policy(raw[policy_mask], y[policy_mask])
    return policy.calibrate(raw), policy


def eval_rows(df, p, policy, mask):
    amt = df.amount_usd.to_numpy()[mask]
    return evaluate(p[mask], df.y.to_numpy()[mask], amt, policy.decide(p[mask], amt),
                    policy.f1_threshold, df.pattern.to_numpy()[mask])


def together(df, tier, model_kind, out_dir: Path):
    cache = config.ARTIFACTS_DIR / "features" / f"{out_dir.name}_together_{tier}.npy"
    t0 = time.time()
    if cache.exists():
        X = np.load(cache)
        log(f"together[{tier}]: features loaded from cache")
    else:
        hubs = find_hubs(df.src.to_numpy(), df.dst.to_numpy(), df.split.to_numpy() == 0)
        X = build_features(df, hubs, TIERS[tier])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, X)
        log(f"together[{tier}]: features {X.shape} built in {time.time() - t0:.0f}s")
    X, scorer, hubs = run_pipeline(df, TIERS[tier], model_kind, X)
    split = df.split.to_numpy()
    p, policy = score_and_policy(df, X, scorer, split == 1)
    log(f"together[{tier}]: trained in {time.time() - t0:.0f}s ({scorer.train_seconds:.0f}s model),"
        f" tree format {scorer.tree_format}")
    bundle = {"tier": tier, "params": TIERS[tier], "scorer": scorer, "policy": policy,
              "hubs": hubs, "n_features": X.shape[1]}
    path = config.ARTIFACTS_DIR / "models" / out_dir.name / f"{tier}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return p, policy, X


def alone_member(df, view, member_owned, tier, model_kind):
    sub = df[view].reset_index(drop=True)
    X, scorer, _ = run_pipeline(sub, TIERS[tier], model_kind)
    split = sub.split.to_numpy()
    owned = member_owned[view]
    p, policy = score_and_policy(sub, X, scorer, (split == 1) & owned)
    test_owned = (split == 2) & owned
    return sub.txn_id.to_numpy()[test_owned], p[test_owned], policy, test_owned, sub


def decision_summary(y, decisions) -> dict:
    """What the release / hold / freeze rule did to the test payments (counts)."""
    y = np.asarray(y) == 1
    d = np.asarray(decisions)
    return {"laundering_stopped": int((y & (d != RELEASE)).sum()), "laundering": int(y.sum()),
            "legit_stopped": int((~y & (d != RELEASE)).sum()),
            "frozen_laundering": int((y & (d == FREEZE)).sum()),
            "frozen_legit": int((~y & (d == FREEZE)).sum())}


def run(dataset, tier, model_kind, seeds, n_members, skip_alone=False):
    out_dir = config.RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pseudonymised(dataset)
    log(f"{dataset}: {len(df):,} rows")
    p_all, policy_all, _ = together(df, tier, model_kind, out_dir)
    test = df.split.to_numpy() == 2
    pred_dir = config.ARTIFACTS_DIR / "predictions" / dataset
    pred_dir.mkdir(parents=True, exist_ok=True)
    amt_all = df.amount_usd.to_numpy()
    np.savez_compressed(pred_dir / f"together_{tier}_{model_kind}.npz", txn_id=np.flatnonzero(test),
                        p=p_all[test], decision=policy_all.decide(p_all[test], amt_all[test]),
                        f1_threshold=policy_all.f1_threshold)

    y_test = df.y.to_numpy()[test]
    decision_counts = {"together": decision_summary(y_test, policy_all.decide(p_all[test], amt_all[test]))}
    results = {"dataset": dataset, "tier": tier, "model": model_kind,
               "split_fractions": config.SPLIT_FRACTIONS, "members": n_members,
               "together_overall": eval_rows(df, p_all, policy_all, test), "seeds": {}}
    log("together overall F1 %.4f" % results["together_overall"]["f1"])

    for seed in ([] if skip_alone else seeds):
        bank_member = assign_members(df, seed, n_members)
        own = owner(df, bank_member)
        seed_res = {"members": []}
        alone_p = np.full(len(df), np.nan)
        alone_dec = np.zeros(len(df), np.int8)
        alone_hit = np.zeros(len(df), bool)
        for m in range(n_members):
            t0 = time.time()
            view = view_mask(df, bank_member, m)
            ids, p, pol, _, _ = alone_member(df, view, own == m, tier, model_kind)
            amt = df.amount_usd.to_numpy()[ids]
            alone_p[ids] = p
            alone_dec[ids] = pol.decide(p, amt)
            alone_hit[ids] = p >= pol.f1_threshold
            mask = test & (own == m)
            assert set(ids) == set(np.flatnonzero(mask)), "alone and together must score the same rows"
            rec = {"member": m, "banks": int(sum(v == m for v in bank_member.values())),
                   "view_rows": int(view.sum()), "owned_test_rows": int(mask.sum()),
                   "alone": evaluate(p, df.y.to_numpy()[ids], amt, alone_dec[ids], pol.f1_threshold,
                                     df.pattern.to_numpy()[ids]),
                   "together": eval_rows(df, p_all, policy_all, mask)}
            seed_res["members"].append(rec)
            log(f"seed {seed} member {m}: view {view.sum():,} rows, test positives "
                f"{rec['alone']['positives']}, F1 alone {rec['alone']['f1']:.4f} vs together "
                f"{rec['together']['f1']:.4f} ({time.time() - t0:.0f}s)")
        # overall alone: every member's own decisions at its own threshold, pooled
        y, amt = df.y.to_numpy()[test], df.amount_usd.to_numpy()[test]
        hit = alone_hit[test]
        ov = f1_at(hit.astype(float), y, 0.5)
        ov["recall_at_review_budget_per_member_mean"] = float(np.mean(
            [r["alone"]["recall_at_review_budget"] for r in seed_res["members"]]))
        ov["laundered_usd_stopped"] = float(amt[(y == 1) & (alone_dec[test] != 0)].sum())
        ov["laundered_usd"] = float(amt[y == 1].sum())
        seed_res["alone_overall"] = ov
        results["seeds"][str(seed)] = seed_res
        decision_counts[f"alone_seed{seed}"] = decision_summary(y, alone_dec[test])
        np.savez_compressed(pred_dir / f"alone_{tier}_{model_kind}_seed{seed}.npz",
                            txn_id=np.flatnonzero(test), p=alone_p[test], decision=alone_dec[test],
                            hit=alone_hit[test], owner=own[test])
        log(f"seed {seed}: overall F1 alone {ov['f1']:.4f} vs together "
            f"{results['together_overall']['f1']:.4f}")

    path = out_dir / f"experiment_{tier}_{model_kind}.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    (out_dir / f"decisions_{tier}_{model_kind}.json").write_text(json.dumps(decision_counts, indent=2))
    (out_dir / f"summary_{tier}_{model_kind}.md").write_text(summarise(results))
    log(f"wrote {path}")
    return results


def summarise(r) -> str:
    t = r["together_overall"]
    lines = [f"# {r['dataset']} — alone vs together ({r['tier']} features, {r['model']})", "",
             f"Together, all test transactions: F1 **{t['f1']:.4f}** (precision {t['precision']:.3f},"
             f" recall {t['recall']:.3f}), recall@0.5% {t['recall_at_review_budget']:.3f}, "
             f"laundered $ stopped {t['laundered_usd_stopped'] / max(t['laundered_usd'], 1):.1%}", ""]
    for seed, s in r["seeds"].items():
        a = s["alone_overall"]
        lines += [f"## Seed {seed}", "",
                  f"Overall alone F1 **{a['f1']:.4f}** vs together **{t['f1']:.4f}**; "
                  f"laundered $ stopped alone {a['laundered_usd_stopped'] / max(a['laundered_usd'], 1):.1%}"
                  f" vs together {t['laundered_usd_stopped'] / max(t['laundered_usd'], 1):.1%}", "",
                  "| Member | Banks | Test positives | F1 alone | F1 together | Recall@0.5% alone | "
                  "Recall@0.5% together | $ stopped alone | $ stopped together |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in s["members"]:
            al, to = m["alone"], m["together"]
            pct = lambda d: d["laundered_usd_stopped"] / max(d["laundered_usd"], 1)
            lines.append(f"| {m['member']} | {m['banks']} | {al['positives']} | {al['f1']:.4f} | "
                         f"{to['f1']:.4f} | {al['recall_at_review_budget']:.3f} | "
                         f"{to['recall_at_review_budget']:.3f} | {pct(al):.1%} | {pct(to):.1%} |")
        lines.append("")
    if r["seeds"]:
        f1s = [s["alone_overall"]["f1"] for s in r["seeds"].values()]
        lines += [f"Alone overall F1 across seeds: mean {np.mean(f1s):.4f}, "
                  f"min {min(f1s):.4f}, max {max(f1s):.4f}", ""]
    lines += ["## Per-pattern recall (together, test split)", "", "| Pattern | Positives | Recall |",
              "|---|---|---|"]
    for k, v in t.get("per_pattern_recall", {}).items():
        lines.append(f"| {k} | {v['positives']} | {v['recall']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="HI-Small")
    ap.add_argument("--tier", choices=list(TIERS), default="full")
    ap.add_argument("--model", choices=["xgb", "snap"], default="xgb")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    ap.add_argument("--members", type=int, default=config.N_MEMBERS)
    ap.add_argument("--together-only", action="store_true")
    a = ap.parse_args()
    run(a.dataset, a.tier, a.model, a.seeds, a.members, skip_alone=a.together_only)


if __name__ == "__main__":
    main()
