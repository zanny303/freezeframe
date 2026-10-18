"""Demo dashboard: a real cross-bank mule chain from the test split, replayed live through
sealed bank clients -> enclave hot path (+ warm tier), with background traffic around it.

The page is the demo operator's view: it merges what each member bank sees on its *own*
decision topic (each bank client unseals only its own messages), plus consortium-level
metrics from the enclave. "Alone" decisions come from the experiment's per-member models.

Run:  arch -x86_64 .venv/bin/python -m demo.app   then open http://localhost:8000
"""
import asyncio
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from freezeframe import config
from freezeframe.consortium import assign_members
from freezeframe.demo_setup import build_engine, chain_candidates, to_payment
from freezeframe.experiment import enclave_key
from freezeframe.pseudonym import Pseudonymizer
from freezeframe.enclave.attest import attestation_doc, attestation_status, code_measurement
from freezeframe.enclave.crypto import ENCLAVE, KeyPair, Rejected, member_id, seal_signed
from freezeframe.enclave.stream import (PAYMENTS, BankClient, HotConsumer, InProcessBus,
                                        decisions_topic, notices_topic)
from freezeframe.ingest import load

DATASET = os.environ.get("FF_DATASET", "HI-Small")
TIER, MODEL = "full", os.environ.get("FF_MODEL", "xgb")
LEAD_SECONDS = 30 * 60        # history ends this long before hop 1
LIVE_LEAD = 6 * 60            # real traffic replayed live before each hop (data seconds)
LIVE_SPEED = float(os.environ.get("FF_LIVE_SPEED", 40))   # x real time while live
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="FreezeFrame demo")
S: dict = {"subscribers": set(), "playing": False, "played": False, "history": [], "wire": None}
REPLAYED = {"hop", "skip", "live", "notice", "play", "done", "security", "attack"}   # re-sent to late pages
ATTACKS = {
    "replay": "Replay a captured bank message",
    "impersonate": "Submit as another bank without its key",
    "impostor": "A real member submits another bank's payment",
    "tamper": "Flip bytes in a sealed message",
    "forge": "Forge a 'release' decision to a bank",
}
REASON_TEXT = {
    "replay": "nonce already seen", "bad_signature": "signature does not verify",
    "undecryptable": "cannot be opened", "not_owner": "payment belongs to another member",
    "malformed": "malformed", "stale": "outside the freshness window",
    "sender_mismatch": "signed by a different member", "unknown_sender": "unknown member",
    "wrong_recipient": "addressed to someone else",
}


def wiretap(topic, msg):
    """The attacker's view of the wire: it can copy sealed payment messages, nothing more."""
    if topic == PAYMENTS:
        S["wire"] = msg


def on_reject(reason, msg):
    broadcast({"type": "security", "who": "enclave", "reason": reason,
               "text": REASON_TEXT.get(reason, reason), "attack": S.pop("pending_attack", None)})


def _predictions(kind):
    path = config.ARTIFACTS_DIR / "predictions" / DATASET / f"{kind}_{TIER}_{MODEL}.npz"
    return dict(np.load(path)) if path.exists() else None


def pick_chain(raw, bank_member):
    """Prefer a chain the pooled model catches at its first hop and a lone bank misses."""
    together = _predictions("together")
    alone_path = config.ARTIFACTS_DIR / "predictions" / DATASET / f"alone_{TIER}_{MODEL}_seed0.npz"
    alone = dict(np.load(alone_path)) if alone_path.exists() else None
    cands = chain_candidates(raw, bank_member, max_span_hours=96)
    if not cands:
        raise RuntimeError("no demo chain found")

    def decisions(arr, ids):
        if arr is None:
            return [None] * len(ids)
        pos = {int(t): i for i, t in enumerate(arr["txn_id"])}
        return [int(arr["decision"][pos[i]]) if i in pos else None for i in ids]

    best = None
    for c in cands[:300]:
        ids = raw.index[raw.attempt == c["attempt"]].tolist()
        tog, alo = decisions(together, ids), decisions(alone, ids)
        caught = sum(1 for d in tog if d) / len(ids)
        missed_alone = sum(1 for d in alo if d == 0) / len(ids)
        score = (bool(tog[0]), caught + missed_alone, c["members"])
        if best is None or score > best[0]:
            best = (score, c, ids, tog, alo)
    _, c, ids, tog, alo = best
    return c, ids, tog, alo


@app.on_event("startup")
async def startup():
    raw = load(DATASET)
    bm = assign_members(raw, 0)
    cand, ids, tog, alo = pick_chain(raw, bm)
    chain = raw.loc[ids].sort_values("ts")
    t_live = int(chain.t.iloc[0]) - LEAD_SECONDS
    engine, keys, warm, ctx = await asyncio.to_thread(
        build_engine, DATASET, 0, 8, 2.0, True, t_live, warm_process=True)
    await asyncio.to_thread(warm.start)
    bus = InProcessBus()
    members = sorted(set(bm.values()))
    # each bank verifies the enclave's attestation document before sealing anything to it
    doc = attestation_doc(keys, {"hot": engine.model_digest})
    banks = {m: BankClient.from_attestation(m, bus, doc, code_measurement(), linger_ms=5.0)
             for m in members}
    consumer = HotConsumer(engine, bus, keys, {m: b.identity for m, b in banks.items()},
                           on_reject=on_reject)
    bus.tap = wiretap
    S.update(engine=engine, warm=warm, keys=keys, bus=bus, banks=banks, consumer=consumer,
             ctx=ctx, chain=chain, cand=cand, bm=bm, tog=tog, alo=alo,
             chain_refs={str(t): i for i, t in enumerate(chain.txn_id)},
             chain_to={})
    for i, r in enumerate(chain.itertuples()):
        S["chain_to"].setdefault((str(r.to_bank), r.to_acct), []).append((i, float(r.amount_usd)))
    asyncio.create_task(consumer.run())
    for b in banks.values():
        asyncio.create_task(b.run_flusher(asyncio.Event()))
    for m in members:
        asyncio.create_task(bank_listener(m))
    asyncio.create_task(stats_pump())


def broadcast(ev: dict):
    if ev["type"] in REPLAYED:
        S["history"].append(ev)
    for q in list(S["subscribers"]):
        q.put_nowait(ev)


async def bank_listener(m):
    """Bank m reads only its own topics and reports what it sees to the operator view."""
    bus, bank = S["bus"], S["banks"][m]
    while True:
        got = False
        msg = bus.get_nowait(decisions_topic(m))
        if msg is not None:
            got = True
            try:
                body = bank.open(msg)
            except Rejected as e:
                broadcast({"type": "security", "who": f"Bank {'ABCDE'[m]}", "reason": e.reason,
                           "text": REASON_TEXT.get(e.reason, e.reason), "attack": "forge"})
                body = {"decisions": [], "rejected": []}
            for d in body["decisions"]:
                hop = S["chain_refs"].get(d["ref"])
                if hop is not None:
                    broadcast({"type": "hop", "hop": hop, "member": m, **d})
            for ref in body.get("rejected", []):
                if ref.startswith("attack-"):
                    broadcast({"type": "security", "who": "enclave", "reason": "not_owner",
                               "text": REASON_TEXT["not_owner"], "attack": "impostor"})
        note = bus.get_nowait(notices_topic(m))
        if note is not None:
            got = True
            for n in bank.open(note):
                # bank m was told to act on one of its own accounts; tie it to the chain hop
                cands = S["chain_to"].get((n["to_bank"], n["to_acct"]), [])
                hop = min(cands, key=lambda c: abs(c[1] - n["amount_usd"]))[0] if cands else None
                if hop is not None:
                    broadcast({"type": "notice", "hop": hop, "member": m, "kind": n["kind"],
                               "to_acct": n["to_acct"][-4:], "amount_usd": n["amount_usd"]})
        if not got:
            await asyncio.sleep(0.02)


async def stats_pump():
    last_n, last_t = 0, time.time()
    while True:
        await asyncio.sleep(0.5)
        eng, cons = S["engine"], S["consumer"]
        now = time.time()
        recent = list(itertools.islice(reversed(cons.e2e_ms), 5000))
        e2e = np.array(recent) if recent else np.array([0.0])
        n = eng.stats.payments
        broadcast({"type": "stats", "rate": (n - last_n) / (now - last_t), "payments": n,
                   "p50": float(np.percentile(e2e, 50)), "p99": float(np.percentile(e2e, 99)),
                   "actions": eng.stats.actions, "upstream_holds": eng.stats.upstream_holds,
                   "escalations": eng.stats.escalations, "playing": S["playing"],
                   "blocked": sum(eng.stats.security.values()) + sum(
                       sum(b.rejected.values()) for b in S["banks"].values())})
        last_n, last_t = n, now


async def _replay(rows, speed):
    """Send bank-side rows through the sealed bank clients at ``speed``x real time."""
    if len(rows) == 0:
        return
    bm = S["bm"]
    rng = np.random.default_rng(int(rows.t.iloc[0]))
    t_data = rows.t.to_numpy().astype(float) + rng.uniform(0, 59, len(rows))
    is_chain = rows.index.isin(S["chain"].index)
    t_data[is_chain] = rows.t.to_numpy()[is_chain] + 30   # hops sit mid-minute, in data order
    order = np.argsort(t_data, kind="stable")
    rows, t_data = rows.iloc[order], t_data[order] - t_data[order][0]
    owners = rows.from_bank.astype(str).map(bm).to_numpy()
    t0 = time.time()
    for i, row in enumerate(rows.itertuples(index=False)):
        due = t0 + t_data[i] / speed
        wait = due - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        await S["banks"][int(owners[i])].send(to_payment(row), sent_at=due)
    for b in S["banks"].values():
        await b.flush()


def _fast_forward(rows):
    """Load rows into the enclave graph without scoring them (the gap between two hops)."""
    if len(rows):
        S["engine"].prime(Pseudonymizer(enclave_key()).apply(rows))


async def play():
    """For each hop: replay LIVE_LEAD of real traffic before it, live, at LIVE_SPEED. The hours
    between hops are fast-forwarded into the graph (not scored, not shown) and labelled so."""
    S["playing"] = True
    chain, live = S["chain"], S["ctx"]["live"]
    hop_t = sorted(set(chain.t.tolist()))
    cursor = int(live.t.iloc[0])
    for k, t_hop in enumerate(hop_t):
        start = max(cursor, t_hop - LIVE_LEAD)
        gap = live[(live.t >= cursor) & (live.t < start)]
        if len(gap):
            broadcast({"type": "skip", "payments": len(gap), "hours": (start - cursor) / 3600})
            await asyncio.to_thread(_fast_forward, gap)
            await asyncio.sleep(0.3)   # let the consumer drain before going live again
        end = t_hop + 60
        broadcast({"type": "live", "hop_time": str(chain[chain.t == t_hop].ts.iloc[0]),
                   "speed": LIVE_SPEED})
        await _replay(live[(live.t >= start) & (live.t < end)], LIVE_SPEED)
        cursor = end
        await asyncio.sleep(1.0)
    await asyncio.sleep(3)
    S["warm"].run_once()
    S["playing"] = False
    broadcast({"type": "done"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/chain")
def api_chain():
    chain, bm = S["chain"], S["bm"]
    acct_tag = {}
    def tag(b, a):
        return acct_tag.setdefault((b, a), f"#{len(acct_tag) + 1}")
    hops = []
    for i, r in enumerate(chain.itertuples()):
        hops.append({"hop": i, "ts": str(r.ts), "from_member": bm[str(r.from_bank)],
                     "to_member": bm[str(r.to_bank)], "from": tag(str(r.from_bank), r.from_acct),
                     "to": tag(str(r.to_bank), r.to_acct), "amount_usd": round(float(r.amount_usd), 2),
                     "currency": str(r.cur_paid), "fmt": str(r.fmt),
                     "together": S["tog"][i], "alone": S["alo"][i]})
    return {"pattern": S["cand"]["pattern"], "span_hours": S["cand"]["span_hours"], "hops": hops,
            "played": S["played"], "dataset": DATASET, "attacks": ATTACKS,
            "measurement": code_measurement()[:12],
            "attestation": attestation_status(), "tree_format": S["engine"].scorer.tree_format}


@app.get("/api/results")
def api_results():
    path = config.RESULTS_DIR / DATASET / f"experiment_{TIER}_{MODEL}.json"
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    seeds = r.get("seeds", {})
    s0 = seeds.get("0") or (next(iter(seeds.values())) if seeds else None)
    pct = lambda d: d["laundered_usd_stopped"] / max(d["laundered_usd"], 1)
    dec_path = config.RESULTS_DIR / DATASET / f"decisions_{TIER}_{MODEL}.json"
    lat_path = config.RESULTS_DIR / DATASET / f"latency_inprocess_{platform.machine()}.json"
    latency = []
    if lat_path.exists():
        latency = [{"speed": x["speed"], "rate": x["achieved_rate_per_s"], "p50": x["e2e_ms"]["p50"],
                    "p99": x["e2e_ms"]["p99"]} for x in json.loads(lat_path.read_text())["runs"]]
    return {
        "decisions": json.loads(dec_path.read_text()) if dec_path.exists() else None,
        "latency": latency,
        "together_f1": r["together_overall"]["f1"],
        "together_usd": pct(r["together_overall"]),
        "alone_f1_by_seed": {k: v["alone_overall"]["f1"] for k, v in seeds.items()},
        "alone_usd_by_seed": {k: pct(v["alone_overall"]) for k, v in seeds.items()},
        "members": [] if not s0 else [{"member": m["member"], "alone": m["alone"]["f1"],
                                       "together": m["together"]["f1"],
                                       "positives": m["alone"]["positives"]} for m in s0["members"]],
    }


@app.post("/api/attack/{kind}")
async def api_attack(kind: str):
    """Run one attack against the live boundary. None of them reaches the graph: each is
    refused by the enclave (or, for a forged decision, by the bank) and shown on the page."""
    if kind not in ATTACKS:
        return {"ok": False, "reason": "unknown attack"}
    bus, banks, keys = S["bus"], S["banks"], S["keys"]
    members = sorted(banks)
    victim = members[0]
    pay = to_payment(S["chain"].iloc[0])
    owner = S["bm"][pay["from_bank"]]
    S["pending_attack"] = kind
    broadcast({"type": "attack", "kind": kind, "label": ATTACKS[kind]})
    if kind == "replay":
        if S["wire"] is None:   # nothing on the wire yet: the bank sends an empty heartbeat batch
            await banks[victim]._publish([], [])
            await asyncio.sleep(0.3)
        await bus.publish(PAYMENTS, S["wire"])
    elif kind == "impersonate":
        mallory = KeyPair()
        await bus.publish(PAYMENTS, {"member": owner, "sealed": seal_signed(
            {"payments": [{**pay, "ref": "attack-impersonate"}], "sent_at": [time.time()]},
            member_id(owner), mallory, ENCLAVE, keys.public_b64)})
    elif kind == "impostor":
        S.pop("pending_attack", None)
        other = next(m for m in members if m != owner)
        await banks[other]._publish([{**pay, "ref": "attack-impostor"}], [time.time()])
    elif kind == "tamper":
        msg = S["wire"] or banks[victim].seal({"payments": [], "sent_at": []})
        t = msg["sealed"]
        await bus.publish(PAYMENTS, {**msg, "sealed": t[:30] + ("A" if t[30] != "A" else "B") + t[31:]})
    elif kind == "forge":
        S.pop("pending_attack", None)
        mallory = KeyPair()
        await bus.publish(decisions_topic(victim), {"sealed": seal_signed(
            {"decisions": [{"ref": pay["ref"], "action": "release", "p": 0.0}], "rejected": []},
            ENCLAVE, mallory, member_id(victim), banks[victim].identity.box_b64)})
    return {"ok": True}


@app.post("/api/play")
async def api_play():
    # One replay per server start: a second one would feed the same payments into the graph again.
    if S["played"]:
        return {"ok": False, "reason": "already played; restart the server to replay"}
    S["played"] = True
    asyncio.create_task(play())
    return {"ok": True}


@app.get("/api/stream")
async def api_stream():
    q: asyncio.Queue = asyncio.Queue()
    for ev in S["history"]:
        q.put_nowait(ev)
    S["subscribers"].add(q)

    async def gen():
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev, default=float)}\n\n"
        finally:
            S["subscribers"].discard(q)
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
