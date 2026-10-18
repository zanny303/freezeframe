"""Streaming latency benchmark: replay the test split through sealed bank clients -> bus ->
enclave hot path (+ warm tier) at several speed-ups of real time.

Arrival times follow the data's own timestamps (minute resolution, spread uniformly inside each
minute), compressed by the speed factor. Latency is measured from each payment's *scheduled*
arrival to the moment its decision is published, so producer stalls count against us. It
includes the full transport security: every batch is Ed25519-signed and sealed by the bank,
verified and replay-checked by the enclave, and every decision is signed and sealed back.

Run:  arch -x86_64 .venv/bin/python -m bench.latency --speeds 1 100 1000 --seconds 60
"""
import argparse
import asyncio
import json
import platform
import time

import numpy as np

from freezeframe import config
from freezeframe.demo_setup import build_engine, to_payment
from freezeframe.enclave.service import attestation_status
from freezeframe.enclave.stream import (BankClient, HotConsumer, InProcessBus, KafkaBus,
                                        decisions_topic)


async def run_speed(engine, warm, keys, live, bank_member, speed, seconds, bus_kind,
                    bank_linger_ms=5.0):
    """Replay ``live`` (bank-side rows, time-sorted, continuing the engine's history) for
    ``seconds`` of wall time at ``speed``x. Returns (result, rows consumed)."""
    bus = InProcessBus() if bus_kind == "inprocess" else KafkaBus(prefix=f"bench{int(time.time())}.")
    members = sorted(set(bank_member.values()))
    banks = {m: BankClient(m, bus, keys.identity, linger_ms=bank_linger_ms) for m in members}
    consumer = HotConsumer(engine, bus, keys, {m: b.identity for m, b in banks.items()})
    engine.stats.stage_ms.clear()

    rng = np.random.default_rng(int(speed))
    t_min = live.t.to_numpy()
    t_data = t_min.astype(float) + rng.uniform(0, 59, len(live))
    order = np.argsort(t_data, kind="stable")
    t_data = t_data[order] - t_data[order][0]
    n = int(np.searchsorted(t_data, seconds * speed))
    # take whole minutes only, so the next speed starts on a fresh timestamp
    if n < len(live):
        last_min = t_min[order][n - 1] if n else t_min[0]
        n = int(np.searchsorted(t_min, last_min, side="right"))
        keep = order < n          # live is time-sorted: rows 0..n-1 are the whole minutes
        order, t_data = order[keep], t_data[keep]
    rows, due = live.iloc[order], t_data / speed

    async def produce():
        t0 = time.time()
        owners = rows.from_bank.astype(str).map(bank_member).to_numpy()
        for i, row in enumerate(rows.itertuples(index=False)):
            wait = t0 + due[i] - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            await banks[int(owners[i])].send(to_payment(row), sent_at=t0 + due[i])
        for b in banks.values():
            await b.flush()
        return time.time() - t0

    task = asyncio.create_task(consumer.run())
    stop_flush = asyncio.Event()
    flushers = [asyncio.create_task(b.run_flusher(stop_flush)) for b in banks.values()]
    wall0 = time.time()
    produce_s = await produce()
    stop_flush.set()
    await asyncio.gather(*flushers)
    while consumer.decided < n and time.time() - wall0 < seconds * 5 + 60:
        await asyncio.sleep(0.05)
    consumer.stop()
    await task
    wall = time.time() - wall0
    if warm._thread is not None:
        warm.run_once()

    decided = 0
    for m in members:
        while True:
            msg = bus.get_nowait(decisions_topic(m))
            if msg is None:
                break
            decided += len(banks[m].open(msg)["decisions"])
    e2e = np.array(consumer.e2e_ms)
    stages = engine.stats.summary()["latency_ms"]
    return {
        "speed": speed, "payments": n, "decided": decided,
        "data_window": [str(rows.ts.min()), str(rows.ts.max())],
        "offered_rate_per_s": n / max(due[-1], 1e-9) if n else 0,
        "achieved_rate_per_s": len(e2e) / wall, "produce_seconds": produce_s,
        "e2e_ms": {f"p{q}": float(np.percentile(e2e, q)) for q in (50, 95, 99)} if len(e2e) else {},
        "e2e_max_ms": float(e2e.max()) if len(e2e) else None,
        "mean_batch": float(np.mean(consumer.batch_sizes)) if consumer.batch_sizes else 0,
        "stages_ms": stages,
        "security_events": engine.stats.summary()["security"],
    }, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HI-Small")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1, 100, 1000])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--offset-hours", type=float, default=6,
                    help="go live this far into the test split; everything before primes the graph")
    ap.add_argument("--bus", choices=["inprocess", "kafka"], default="inprocess")
    ap.add_argument("--budget-ms", type=float, default=50)
    ap.add_argument("--warm-interval", type=float, default=3.0,
                    help="seconds between warm-tier runs; 0 disables the warm tier")
    ap.add_argument("--warm-thread", action="store_true",
                    help="run the warm tier as a thread in the enclave process (default: own process)")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    ap.add_argument("--bank-linger-ms", type=float, default=5.0,
                    help="bank gateways batch payments this long before sealing (counted in latency)")
    a = ap.parse_args()

    from freezeframe.ingest import load
    raw = load(a.dataset)
    t_live = int(raw[raw.split == 2].t.iloc[0] + a.offset_hours * 3600)
    print(f"priming enclave with history before {raw.ts.min() + np.timedelta64(t_live, 's')} ...",
          flush=True)
    t0 = time.time()
    engine, keys, warm, ctx = build_engine(a.dataset, live_from_t=t_live,
                                           warm_interval=a.warm_interval or 3.0,
                                           prime_warm=a.warm_interval > 0,
                                           warm_process=a.warm_interval > 0 and not a.warm_thread)
    print(f"primed in {time.time() - t0:.0f}s", flush=True)
    if a.warm_interval > 0:
        warm.start()
    live, bm = ctx["live"], ctx["bank_member"]
    runs = []
    for s in a.speeds:
        r, used = asyncio.run(run_speed(engine, warm, keys, live, bm, s, a.seconds, a.bus,
                                        a.bank_linger_ms))
        live = live.iloc[used:]
        runs.append(r)
        print(json.dumps({k: r[k] for k in ("speed", "payments", "achieved_rate_per_s", "e2e_ms",
                                             "mean_batch")}), flush=True)
    if warm._thread is not None:
        warm.stop()
    within = [r for r in runs if r["e2e_ms"] and r["e2e_ms"]["p99"] <= a.budget_ms]
    out = {
        "dataset": a.dataset, "bus": a.bus, "budget_ms": a.budget_ms,
        "bank_linger_ms": a.bank_linger_ms, "warm_interval_s": a.warm_interval,
        "warm_mode": "off" if a.warm_interval <= 0 else ("thread" if a.warm_thread else "process"),
        "host": {"machine": platform.machine(), "platform": platform.platform(),
                 "processor": platform.processor()},
        "scoring_tree_format": engine.scorer.tree_format,
        "attestation": attestation_status(),
        "runs": runs,
        "max_rate_within_budget_per_s": max((r["achieved_rate_per_s"] for r in within), default=None),
        "engine": engine.stats.summary(), "warm": warm.summary(),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / a.dataset / f"latency_{a.bus}_{platform.machine()}{a.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
