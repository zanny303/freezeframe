"""Enclave boundary: owners only, no raw accounts inside, sealed transport, chain following."""
import asyncio
import copy
import json
import pickle

from fastapi.testclient import TestClient

from freezeframe.consortium import assign_members
from freezeframe.demo_setup import to_payment
from freezeframe.enclave.crypto import ENCLAVE, KeyPair, member_id, open_signed, seal_signed
from freezeframe.enclave.engine import Engine
from freezeframe.enclave.service import create_app
from freezeframe.enclave.stream import BankClient, HotConsumer, InProcessBus, decisions_topic
from freezeframe.enclave.warm import WarmTier

from .conftest import KEY


def _engine(raw_df, pseudo_df, bundles, seed=0):
    bm = assign_members(raw_df, seed)
    eng = Engine(copy.deepcopy(bundles["hot"]), bm, KEY, {"US Dollar": 1.0}, t0=raw_df.ts.min(), threads=2)
    eng.prime(pseudo_df[pseudo_df.split < 2])
    return eng, bm


def _live(raw_df):
    return raw_df[raw_df.split == 2]


def test_decisions_only_to_owner(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    live = _live(raw_df).head(400)
    items = [(bm[str(r.from_bank)], to_payment(r)) for r in live.itertuples()]
    out, rejected = eng.process(items)
    assert not rejected
    owner_of = {to_payment(r)["ref"]: bm[str(r.from_bank)] for r in live.itertuples()}
    for m, decs in out.items():
        assert all(owner_of[d["ref"]] == m for d in decs)
    assert sum(len(d) for d in out.values()) == len(live)


def test_cannot_submit_another_members_payment(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    r = next(_live(raw_df).itertuples())
    imposter = (bm[str(r.from_bank)] + 1) % 5
    out, rejected = eng.process([(imposter, to_payment(r))])
    assert out == {} and rejected == {imposter: [str(r.txn_id)]}


def test_no_raw_account_inside_enclave(raw_df, pseudo_df, bundles, tmp_path):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    warm = WarmTier(eng, bundles["full"], threads=1)
    live = _live(raw_df).head(300)
    eng.process([(bm[str(r.from_bank)], to_payment(r)) for r in live.itertuples()])
    queued = list(eng.warm_queue)
    warm.run_once()
    raw_accounts = set(raw_df.from_acct) | set(raw_df.to_acct)
    # features, queued frames for the warm tier, metrics and the model artefacts
    for cols, _ in queued:
        assert not {"from_acct", "to_acct"} & set(cols)
        values = {str(v) for col in cols.values() for v in col}
        assert not raw_accounts & values
    blob = json.dumps(eng.stats.summary()) + str(pickle.dumps(bundles["hot"]))
    assert not any(a in blob for a in list(raw_accounts)[:200])


def test_freeze_notice_names_only_receivers_own_account(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    eng.policy.freeze_threshold = 0.0          # force freezes
    live = _live(raw_df).head(50)
    eng.process([(bm[str(r.from_bank)], to_payment(r)) for r in live.itertuples()])
    for m in range(5):
        for n in eng.drain_notices(m):
            assert bm[n["to_bank"]] == m
            assert set(n) == {"case", "kind", "to_bank", "to_acct", "amount_usd", "ts"}


def test_hop_two_is_held(raw_df, pseudo_df, bundles):
    """Flagging hop 1 puts its receiver on the watchlist; hop 2 from that account is held."""
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    base = to_payment(next(_live(raw_df).itertuples()))
    hop1 = {**base, "ref": "h1", "to_bank": "001", "to_acct": "MULE1", "amount_paid": 1e9}
    hop2 = {**base, "ref": "h2", "from_bank": "001", "from_acct": "MULE1", "to_acct": "MULE2",
            "amount_paid": 10.0}
    eng.policy.review_cost_usd = 1e-12          # hop 1 is held on expected loss
    out1, _ = eng.process([(bm[base["from_bank"]], hop1)])
    assert out1[bm[base["from_bank"]]][0]["action"] != "release"
    eng.policy.review_cost_usd = 1e12           # hop 2 alone would be released
    out2, _ = eng.process([(bm["001"], hop2)])
    d = out2[bm["001"]][0]
    assert d["action"] == "hold" and d["reason"] == "upstream_flag"


def test_sealed_stream_roundtrip(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    keys = KeyPair()

    async def go():
        bus = InProcessBus()
        banks = {m: BankClient(m, bus, keys.identity) for m in range(5)}
        cons = HotConsumer(eng, bus, keys, {m: b.identity for m, b in banks.items()})
        task = asyncio.create_task(cons.run())
        live = _live(raw_df).head(120)
        for r in live.itertuples():
            await banks[bm[str(r.from_bank)]].send(to_payment(r))
        while cons.decided < len(live):
            await asyncio.sleep(0.01)
        cons.stop()
        await task
        got = {}
        for m, b in banks.items():
            while (msg := bus.get_nowait(decisions_topic(m))) is not None:
                for d in b.open(msg)["decisions"]:
                    got[d["ref"]] = m
                other = banks[(m + 1) % 5]
                try:
                    other.open(msg)
                    raise AssertionError("another member could read the decisions")
                except Exception as e:
                    assert not isinstance(e, AssertionError)
        return live, got

    live, got = asyncio.run(go())
    assert set(got) == set(live.txn_id.astype(str))
    assert all(got[str(r.txn_id)] == bm[str(r.from_bank)] for r in live.itertuples())


def test_http_service(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    keys = KeyPair()
    app = create_app(eng, keys)
    client = TestClient(app)
    info = client.get("/v1/enclave").json()
    assert info["attestation"] in ("simulated", "secure-execution")
    assert client.get("/v1/health").json()["state"] == "ready"
    bank = KeyPair()
    r = next(_live(raw_df).itertuples())
    m = bm[str(r.from_bank)]
    sealed = seal_signed({"payments": [to_payment(r)]}, member_id(m), bank, ENCLAVE, info["public_key"])
    assert client.post("/v1/submit", json={"member": m, "sealed": sealed}).status_code == 403
    token = app.state.enrollment_tokens[m]
    enroll = {"box_public_key": bank.public_b64, "verify_key": bank.verify_b64, "token": token}
    assert client.post(f"/v1/members/{m}/enroll", json=enroll).status_code == 200
    sealed = seal_signed({"payments": [to_payment(r)]}, member_id(m), bank, ENCLAVE, info["public_key"])
    resp = client.post("/v1/submit", json={"member": m, "sealed": sealed}).json()
    _, body = open_signed(resp["sealed"], bank, member_id(m), info["verify_key"], None,
                          expect_from=ENCLAVE)
    assert [d["ref"] for d in body["decisions"]] == [str(r.txn_id)]
    metrics = client.get("/v1/metrics").json()
    assert metrics["payments"] == 1 and "latency_ms" in metrics


def test_model_hot_swap(raw_df, pseudo_df, bundles):
    eng, bm = _engine(raw_df, pseudo_df, bundles)
    r = next(_live(raw_df).itertuples())
    m = bm[str(r.from_bank)]
    new = {**copy.deepcopy(bundles["hot"]), "version": "hot-v2"}
    eng.swap_model(new)
    out, _ = eng.process([(m, to_payment(r))])
    assert out[m][0]["model"] == "hot-v2"
    try:
        eng.swap_model(copy.deepcopy(bundles["full"]))
        raise AssertionError("swap with different graph features must fail")
    except ValueError:
        pass


def _orphan_probe(parent_pid, out):
    """Child of a short-lived parent: arms the watchdog, then reports its pid and waits."""
    import os
    import time
    from freezeframe.enclave.warm import _exit_with_parent
    _exit_with_parent(parent_pid, every_s=0.2)
    out.put(os.getpid())
    time.sleep(60)


def _spawn_probe_then_die(out):
    import multiprocessing as mp
    import os
    p = mp.get_context("spawn").Process(target=_orphan_probe, args=(os.getpid(), out))
    p.start()
    os._exit(0)          # die abruptly, like a killed server: no cleanup of children


def test_warm_worker_exits_when_parent_dies():
    import multiprocessing as mp
    import os
    import time
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    parent = ctx.Process(target=_spawn_probe_then_die, args=(out,))
    parent.start()
    child = out.get(timeout=60)
    parent.join(timeout=10)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    os.kill(child, 9)
    raise AssertionError("orphaned worker kept running after its parent died")
