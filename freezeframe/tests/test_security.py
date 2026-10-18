"""Attacks on the enclave boundary: each must be refused, counted, and must not stop the enclave."""
import asyncio
import copy
import time

import pytest
from fastapi.testclient import TestClient

from freezeframe.consortium import assign_members
from freezeframe.demo_setup import to_payment
from freezeframe.enclave.attest import attestation_doc, code_measurement, verify_attestation
from freezeframe.enclave.crypto import (ENCLAVE, KeyPair, Rejected, ReplayGuard, member_id,
                                        open_signed, seal_signed)
from freezeframe.enclave.engine import WATCH_PRUNE_EVERY, Engine
from freezeframe.enclave.identity import Registry
from freezeframe.enclave.service import create_app
from freezeframe.enclave.stream import (PAYMENTS, BankClient, HotConsumer, InProcessBus,
                                        decisions_topic)
from freezeframe.explain import LABELS

from .conftest import KEY


def _engine(raw_df, pseudo_df, bundles, seed=0):
    bm = assign_members(raw_df, seed)
    eng = Engine(copy.deepcopy(bundles["hot"]), bm, KEY, {"US Dollar": 1.0}, t0=raw_df.ts.min(), threads=2)
    eng.prime(pseudo_df[pseudo_df.split < 2])
    return eng, bm


def _live(raw_df):
    return raw_df[raw_df.split == 2]


# ---- envelope -------------------------------------------------------------------------------

def _open(token, enclave, reg, guard, m):
    return open_signed(token, enclave, ENCLAVE, reg.verify_key_for, guard, expect_from=member_id(m))


def test_envelope_checks():
    enclave, alice, mallory = KeyPair(), KeyPair(), KeyPair()
    reg = Registry.of({0: alice.identity, 1: mallory.identity})
    guard = ReplayGuard()
    ok = seal_signed({"x": 1}, member_id(0), alice, ENCLAVE, enclave.public_b64)
    assert _open(ok, enclave, reg, guard, 0) == (member_id(0), {"x": 1})

    def reason(token, m=0, **kw):
        with pytest.raises(Rejected) as e:
            open_signed(token, enclave, ENCLAVE, reg.verify_key_for, guard,
                        expect_from=member_id(m), **kw)
        return e.value.reason

    assert reason(ok) == "replay"                                         # captured and resent
    forged = seal_signed({"x": 2}, member_id(0), mallory, ENCLAVE, enclave.public_b64)
    assert reason(forged) == "bad_signature"                              # claims to be member 0
    other = seal_signed({"x": 3}, member_id(1), mallory, ENCLAVE, enclave.public_b64)
    assert reason(other, m=0) == "sender_mismatch"                        # routed as member 0
    stale = seal_signed({"x": 4}, member_id(0), alice, ENCLAVE, enclave.public_b64,
                        issued_at=time.time() - 3600)
    assert reason(stale) == "stale"
    elsewhere = seal_signed({"x": 5}, member_id(0), alice, member_id(1), enclave.public_b64)
    assert reason(elsewhere) == "wrong_recipient"                         # forwarded envelope
    tampered = ok[:40] + ("A" if ok[40] != "A" else "B") + ok[41:]
    assert reason(tampered) == "undecryptable"
    assert reason("not even base64 !!") == "undecryptable"
    unknown = seal_signed({"x": 6}, member_id(7), KeyPair(), ENCLAVE, enclave.public_b64)
    assert reason(unknown, m=7) == "unknown_sender"


def test_replay_guard_bounded():
    g = ReplayGuard(skew_s=1.0)
    assert g.check("a", "n1", now=0) and not g.check("a", "n1", now=0.5)
    for i in range(100):
        g.check("a", f"x{i}", now=1.0)
    g.check("a", "late", now=10.0)                   # everything older than 2 * skew is dropped
    assert len(g) == 1


# ---- stream ---------------------------------------------------------------------------------

def test_consumer_survives_attacks_and_counts_them(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    keys = KeyPair()
    rejected = []

    async def go():
        bus = InProcessBus()
        captured = []
        bus.tap = lambda topic, msg: captured.append(msg) if topic == PAYMENTS else None
        banks = {m: BankClient(m, bus, keys.identity) for m in range(5)}
        cons = HotConsumer(eng, bus, keys, {m: b.identity for m, b in banks.items()},
                           on_reject=lambda reason, msg: rejected.append(reason))
        task = asyncio.create_task(cons.run())
        live = _live(raw_df).head(40)
        rows = list(live.itertuples())
        r0 = rows[0]
        owner = bm[str(r0.from_bank)]
        imposter = (owner + 1) % 5
        await banks[owner].send(to_payment(r0))
        while cons.decided < 1:
            await asyncio.sleep(0.01)
        mallory = KeyPair()
        attacks = [
            {"junk": True},                                                  # malformed
            {"member": owner, "sealed": "garbage"},                          # undecryptable
            captured[0],                                                     # replay
            {"member": owner, "sealed": seal_signed(                         # forged signature
                {"payments": [to_payment(rows[1])], "sent_at": [time.time()]},
                member_id(owner), mallory, ENCLAVE, keys.public_b64)},
        ]
        for a in attacks:
            await bus.publish(PAYMENTS, a)
        # an authenticated member submitting someone else's payment, plus a malformed payment
        await banks[imposter].send(to_payment(rows[2]))
        await banks[owner].send({"ref": "bad", "ts": "not a time"})
        for r in rows[3:]:
            await banks[bm[str(r.from_bank)]].send(to_payment(r))
        while cons.decided < len(rows):
            await asyncio.sleep(0.01)
        cons.stop()
        await task
        # forged decision on the owner's topic: the bank refuses it
        fake = {"sealed": seal_signed({"decisions": [{"ref": str(r0.txn_id), "action": "release"}]},
                                      ENCLAVE, mallory, member_id(owner), banks[owner].identity.box_b64)}
        with pytest.raises(Rejected):
            banks[owner].open(fake)
        decided = {}
        for m, b in banks.items():
            while (msg := bus.get_nowait(decisions_topic(m))) is not None:
                body = b.open(msg)
                decided.update({d["ref"]: d for d in body["decisions"]})
                if m == imposter:
                    assert str(rows[2].txn_id) in body["rejected"] or not body["rejected"]
        return rows, decided

    rows, decided = asyncio.run(go())
    sec = eng.stats.summary()["security"]
    assert sec["malformed"] >= 2 and sec["undecryptable"] == 1
    assert sec["replay"] == 1 and sec["bad_signature"] == 1 and sec["not_owner"] == 1
    assert sorted(rejected) == ["bad_signature", "malformed", "replay", "undecryptable"]
    assert str(rows[2].txn_id) not in decided                    # imposter's payment refused
    assert all(str(r.txn_id) in decided for r in rows[3:])       # everyone else still served


# ---- HTTP -----------------------------------------------------------------------------------

def test_http_enrollment_and_notices_are_authenticated(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    keys = KeyPair()
    app = create_app(eng, keys)
    c = TestClient(app)
    info = c.get("/v1/enclave").json()
    tokens = app.state.enrollment_tokens
    alice, mallory = KeyPair(), KeyPair()
    body = lambda k, t: {"box_public_key": k.public_b64, "verify_key": k.verify_b64, "token": t}
    assert c.post("/v1/members/0/enroll", json=body(mallory, "guess")).status_code == 403
    assert c.post("/v1/members/0/enroll", json=body(alice, tokens[0])).status_code == 200
    # the token is spent: nobody can overwrite member 0's key with it
    assert c.post("/v1/members/0/enroll", json=body(mallory, tokens[0])).status_code == 403
    # notices can't be drained by someone who isn't member 0
    req = seal_signed({"op": "notices"}, member_id(0), mallory, ENCLAVE, info["public_key"])
    assert c.post("/v1/notices", json={"member": 0, "sealed": req}).status_code == 401
    req = seal_signed({"op": "notices"}, member_id(0), alice, ENCLAVE, info["public_key"])
    r = c.post("/v1/notices", json={"member": 0, "sealed": req})
    assert r.status_code == 200
    assert c.post("/v1/notices", json={"member": 0, "sealed": req}).status_code == 401   # replay
    _, notes = open_signed(r.json()["sealed"], alice, member_id(0), info["verify_key"], None)
    assert notes == []
    # key rotation needs the current key
    new = KeyPair()
    rot = {"box_public_key": new.public_b64, "verify_key": new.verify_b64}
    bad = seal_signed(rot, member_id(0), mallory, ENCLAVE, info["public_key"])
    assert c.post("/v1/members/0/rotate", json={"sealed": bad}).status_code == 401
    good = seal_signed(rot, member_id(0), alice, ENCLAVE, info["public_key"])
    assert c.post("/v1/members/0/rotate", json={"sealed": good}).status_code == 200
    assert app.state.registry.get(0).verify_b64 == new.verify_b64
    sec = c.get("/v1/metrics").json()["security"]
    assert sec["unknown_sender"] == 2 and sec["bad_signature"] == 2 and sec["replay"] == 1


def test_attestation_document():
    keys = KeyPair()
    doc = attestation_doc(keys, {"hot": "abc"})
    ident = verify_attestation(doc, expected_measurement=code_measurement(),
                               expected_models={"hot": "abc"})
    assert ident == keys.identity
    assert doc["status"] in ("simulated", "secure-execution")
    for change in ({"code_measurement": "0" * 64}, {"enclave_box_key": KeyPair().public_b64},
                   {"models": {"hot": "evil"}}):
        with pytest.raises(Rejected):
            verify_attestation({**doc, **change})
    with pytest.raises(Rejected):
        verify_attestation(doc, expected_measurement="f" * 64)
    with pytest.raises(Rejected):
        verify_attestation(doc, expected_models={"hot": "other"})
    if doc["status"] == "simulated":
        with pytest.raises(Rejected):
            verify_attestation(doc, allow_simulated=False)
    bank = BankClient.from_attestation(0, InProcessBus(), doc, code_measurement())
    assert bank.enclave == keys.identity


# ---- engine: watchlist, validation, reason codes --------------------------------------------

def test_hub_receivers_are_not_watched_and_watchlist_is_pruned(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    base = to_payment(next(_live(raw_df).itertuples()))
    hub_pid = int(eng.hubs[0])
    eng.policy.freeze_threshold = 0.0                       # flag everything
    # find the raw account behind the hub pseudonym in the synthetic data
    hub = next((b, a) for b, a in zip(raw_df.to_bank.astype(str), raw_df.to_acct)
               if eng.pseudo.one(b, a) == hub_pid)
    eng.process([(bm[base["from_bank"]], {**base, "ref": "to-hub", "to_bank": hub[0], "to_acct": hub[1]})])
    assert hub_pid not in eng.watch and eng.stats.watch_skipped_hub == 1
    eng.watch[123] = 0                                       # long expired
    eng._since_prune = WATCH_PRUNE_EVERY
    eng.process([(bm[base["from_bank"]], {**base, "ref": "tick"})])
    assert 123 not in eng.watch


def test_malformed_payment_does_not_sink_the_batch(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    rows = list(_live(raw_df).head(5).itertuples())
    m = bm[str(rows[0].from_bank)]
    good = [to_payment(r) for r in rows if bm[str(r.from_bank)] == m]
    bad = [{"ref": "b1"}, {**good[0], "ref": "b2", "amount_paid": float("nan")},
           {**good[0], "ref": "b3", "ts": "yesterday"}]
    out, rejected = eng.process([(m, p) for p in good + bad])
    assert [d["ref"] for d in out[m]] == [p["ref"] for p in good]
    assert sorted(rejected[m]) == ["b1", "b2", "b3"]
    assert eng.stats.security["malformed"] == 3


def test_flagged_payments_carry_privacy_safe_reasons(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    assert eng.explain
    eng.policy.freeze_threshold = 0.0                       # flag everything
    live = _live(raw_df).head(60)
    out, _ = eng.process([(bm[str(r.from_bank)], to_payment(r)) for r in live.itertuples()])
    decisions = [d for ds in out.values() for d in ds]
    flagged = [d for d in decisions if d["action"] != "release"]
    assert flagged and any(d.get("why") for d in flagged)
    raw_accounts = set(raw_df.from_acct) | set(raw_df.to_acct)
    for d in flagged:
        for w in d.get("why", []):
            assert w["code"] in LABELS and 0 < w["share"] <= 1
            assert not any(a in str(w) for a in raw_accounts)
    assert all("why" not in d for d in decisions if d["action"] == "release")
    assert "explain" in eng.stats.summary()["latency_ms"]
