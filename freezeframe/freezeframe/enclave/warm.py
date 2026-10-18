"""Warm tier: every few seconds, re-score recent payments with the full feature set.

The hot path decides with cheaper features. The warm tier runs its own GFP over the same
payments with every pattern (including scatter-gather) and the full model. When a payment the
hot path released now looks like laundering, it escalates: the receiving account is watched
(so its next outgoing payment is held at hop 2) and the receiving bank gets a hold notice.
"""
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from ..decide import RELEASE
from ..features import GraphFeatures, edge_matrix, raw_features


class WarmTier:
    def __init__(self, engine, bundle: dict, interval_s: float = 3.0, threads: int = 4):
        from .attest import bundle_digest
        self.engine = engine
        self.scorer, self.policy, self.hubs = bundle["scorer"], bundle["policy"], bundle["hubs"]
        self.model_digest = bundle_digest(bundle)
        self.graph = GraphFeatures(bundle["params"], threads=threads)
        self.interval_s = interval_s
        self.runs = 0
        self.rescored = 0
        self.escalated = 0
        self.last_run_ms = 0.0
        self._stop = threading.Event()
        self._thread = None

    def prime(self, df: pd.DataFrame):
        self.graph.stream(edge_matrix(df, self.hubs))

    def run_once(self) -> int:
        batches = []
        while self.engine.warm_queue:
            batches.append(self.engine.warm_queue.popleft())
        if not batches:
            return 0
        t0 = time.perf_counter()
        df = pd.concat([pd.DataFrame(b[0]) for b in batches], ignore_index=True)
        hot = np.concatenate([b[1] for b in batches])
        order = np.argsort(df.t.to_numpy(), kind="stable")
        df, hot = df.iloc[order].reset_index(drop=True), hot[order]
        graph = self.graph.stream(edge_matrix(df, self.hubs))
        X = np.hstack([raw_features(df, self.hubs), graph])
        p = self.policy.calibrate(self.scorer.predict(X, n_jobs=2))
        full = self.policy.decide(p, df.amount_usd.to_numpy())
        n = 0
        for i in np.flatnonzero((hot == RELEASE) & (full != RELEASE)):
            self.engine.escalate(int(df.txn_id.iat[i]), int(df.dst.iat[i]), int(df.t.iat[i]),
                                 float(df.amount_usd.iat[i]), str(df.ts.iat[i]))
            n += 1
        self.runs += 1
        self.rescored += len(df)
        self.escalated += n
        self.last_run_ms = (time.perf_counter() - t0) * 1e3
        return n

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self.run_once()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="warm-tier")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        self.run_once()

    def summary(self) -> dict:
        return {"runs": self.runs, "rescored": self.rescored, "escalated": self.escalated,
                "last_run_ms": self.last_run_ms, "interval_s": self.interval_s}


# ---- process mode ---------------------------------------------------------------------------
# In a thread the warm tier's Python work holds the GIL for hundreds of milliseconds per run,
# which stalls the hot path (bench: p99 230 ms vs 20 ms at 100x). In process mode it runs in its
# own interpreter: the engine ships each batch's pseudonymous columns over a queue and applies
# the escalations that come back at the start of its next batch.

_WARM_COLS = ("txn_id", "t", "ts", "from_bank", "to_bank", "fmt", "cur_paid", "cur_recv",
              "amount_usd", "src", "dst")


def _exit_with_parent(parent_pid: int, every_s: float = 1.0):
    """Exit this worker as soon as the enclave process that spawned it is gone.

    ``daemon=True`` only covers a clean interpreter shutdown of the parent; a parent killed by
    a signal leaves the worker orphaned (re-parented to init/launchd) and still holding the
    graph in memory. Started before priming, so it covers that phase too.
    """
    import os

    def watch():
        while True:
            time.sleep(every_s)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()


def _warm_worker(dataset, t_live, interval_s, threads, inbox, outbox, parent_pid=None):
    import queue as _q

    if parent_pid is not None:
        _exit_with_parent(parent_pid)

    from ..demo_setup import load_bundle
    from ..experiment import enclave_key
    from ..ingest import load
    from ..pseudonym import Pseudonymizer

    class _Shim:  # stands in for the engine inside the worker
        def __init__(self):
            self.warm_queue = deque()
            self.found = []

        def escalate(self, txn_id, dst_pid, t, amount_usd, ts):
            self.found.append((txn_id, dst_pid, t, amount_usd, ts))

    shim = _Shim()
    tier = WarmTier(shim, load_bundle(dataset, "full"), interval_s=interval_s, threads=threads)
    raw = load(dataset)
    hist = Pseudonymizer(enclave_key()).apply(raw[raw.t < t_live])
    del raw
    tier.prime(hist)
    del hist
    outbox.put(("ready", None))
    stopping = False
    next_run = time.monotonic() + interval_s
    while True:
        try:
            item = inbox.get(timeout=max(0.0, next_run - time.monotonic()))
            if item is None:
                stopping = True
            else:
                shim.warm_queue.append(item)
                if time.monotonic() < next_run:
                    continue
        except _q.Empty:
            pass
        tier.run_once()
        if shim.found:
            outbox.put(("escalations", shim.found))
            shim.found = []
        outbox.put(("summary", tier.summary()))
        next_run = time.monotonic() + interval_s
        if stopping:
            outbox.put(("stopped", None))
            return


class WarmProcess:
    """WarmTier in a separate process, primed from the same history as the engine."""

    def __init__(self, engine, dataset: str, t_live: int, interval_s: float = 3.0, threads: int = 4):
        import multiprocessing as mp
        import os
        from ..demo_setup import bundle_file_digest
        ctx = mp.get_context("spawn")
        self.engine = engine
        self.model_digest = bundle_file_digest(dataset, "full")
        self.inbox, self.outbox = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_warm_worker, daemon=True, name="warm-tier",
                                args=(dataset, t_live, interval_s, threads, self.inbox, self.outbox,
                                      os.getpid()))
        self._summary = {"runs": 0, "rescored": 0, "escalated": 0, "interval_s": interval_s,
                         "mode": "process"}
        self._thread = None
        engine.warm_sink = self._ship
        engine.warm_poll = self.poll

    def _ship(self, cols, actions):
        self.inbox.put(({k: cols[k] for k in _WARM_COLS}, actions))

    def poll(self):
        """Apply whatever the worker has sent back; never blocks."""
        import queue as _q
        while True:
            try:
                kind, body = self.outbox.get_nowait()
            except _q.Empty:
                return
            self._handle(kind, body)

    def _handle(self, kind, body):
        if kind == "escalations":
            for txn_id, dst, t, amount, ts in body:
                self.engine.escalate(int(txn_id), int(dst), int(t), float(amount), str(ts))
        elif kind == "summary":
            self._summary.update(body, mode="process")

    def start(self, wait_ready: bool = True):
        self.proc.start()
        self._thread = True
        if wait_ready:
            while True:
                kind, body = self.outbox.get()
                if kind == "ready":
                    return
                self._handle(kind, body)

    def run_once(self):
        """Drain what has arrived so far (the worker runs on its own schedule)."""
        self.poll()

    def stop(self):
        self.inbox.put(None)
        while True:
            kind, body = self.outbox.get(timeout=120)
            if kind == "stopped":
                break
            self._handle(kind, body)
        self.proc.join(timeout=30)

    def summary(self) -> dict:
        return dict(self._summary)
