import numpy as np

from freezeframe.consortium import assign_members, owner, receiver, view_mask
from freezeframe.decide import FREEZE, HOLD, RELEASE, best_f1_threshold, fit_policy, precision_threshold
from freezeframe.metrics import evaluate, recall_at_fraction
from freezeframe.pseudonym import Pseudonymizer

from .conftest import KEY


def test_pseudonyms_keyed_and_stable(raw_df):
    a = Pseudonymizer(KEY).column(raw_df.from_bank, raw_df.from_acct)
    b = Pseudonymizer(KEY).column(raw_df.from_bank, raw_df.from_acct)
    c = Pseudonymizer(b"another-key-0123456789abcdef-xyz").column(raw_df.from_bank, raw_df.from_acct)
    assert (a == b).all() and (a != c).mean() > 0.99
    assert a.max() < 2 ** 52


def test_bank_is_part_of_account_identity():
    p = Pseudonymizer(KEY)
    assert p.one("001", "8000A") != p.one("002", "8000A")


def test_apply_drops_raw_accounts(raw_df):
    out = Pseudonymizer(KEY).apply(raw_df)
    assert "from_acct" not in out and "to_acct" not in out
    raw = set(raw_df.from_acct) | set(raw_df.to_acct)
    assert not raw & set(out.astype(str).to_numpy().ravel())


def test_views_and_owners(raw_df):
    bm = assign_members(raw_df, seed=3)
    own, rec = owner(raw_df, bm), receiver(raw_df, bm)
    for m in range(5):
        v = view_mask(raw_df, bm, m)
        assert ((own == m) | (rec == m) == v).all()
        assert v[own == m].all()            # a member sees everything it owns
    assert assign_members(raw_df, 3) == bm and assign_members(raw_df, 4) != bm


def test_thresholds():
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    y = np.array([1, 1, 0, 1, 0, 0])
    assert best_f1_threshold(p, y) in (0.8, 0.2)
    assert precision_threshold(p, y, target=0.95, min_tp=1) == 0.8
    assert precision_threshold(p, y, target=0.95, min_tp=5) == float("inf")


def test_decision_rule():
    rng = np.random.default_rng(0)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw ** 4).astype(int)
    pol = fit_policy(raw, y)
    pol.freeze_threshold = 0.9
    d = pol.decide(np.array([0.95, 0.5, 0.001, 0.001]), np.array([10.0, 1000.0, 1e6, 10.0]))
    assert d.tolist() == [FREEZE, HOLD, HOLD, RELEASE]


def test_metrics():
    p = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([1, 0, 0, 1])
    amt = np.array([100.0, 5.0, 7.0, 50.0])
    m = evaluate(p, y, amt, np.array([2, 0, 1, 0]), 0.5, np.array(["CYCLE", "", "", ""]))
    assert m["tp"] == 1 and m["fp"] == 1 and m["laundered_usd_stopped"] == 100.0
    assert m["per_pattern_recall"]["CYCLE"]["recall"] == 1.0
    assert recall_at_fraction(p, y, 0.25) == 0.5
