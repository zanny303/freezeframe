"""The enclave's scoring core, shared by the HTTP service, the stream consumer and the benchmark.

Per micro-batch: check ownership -> pseudonymise -> GFP update (one call per timestamp, as in
training) -> hot booster -> calibrate -> decide -> route each result to its owner only.

A micro-batch may mix payments from several members: the graph must see every member's
payments in one time order, so they are processed together and split again on the way out.

Chain following: when a payment is held or frozen, its receiving account goes on a watchlist,
so outgoing payments from that account in the next WATCH_SECONDS (hop 2 of a mule chain) are
held even if their own score is low. The warm tier (warm.py) adds accounts to the same
watchlist. Hub accounts (very high degree, e.g. a payment processor) are never watched: one
held payment into a hub would otherwise hold every payment the hub makes for three days.

Inputs are untrusted even from an authenticated member: a payment with missing or malformed
fields is rejected (reason ``malformed``) instead of failing the whole micro-batch.

Held and frozen payments carry privacy-safe reason codes (explain.py): which *families* of
evidence drove the score (cycle, fan-in/out, amount, ...), never another bank's data.
"""
import math
import secrets
import threading
from datetime import datetime
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..decide import ACTIONS, FREEZE, HOLD, RELEASE
from ..explain import UPSTREAM, reason_codes
from ..features import GraphFeatures, edge_arrays, edge_matrix, feature_groups, raw_features_live
from ..pseudonym import Pseudonymizer

WATCH_SECONDS = 72 * 3600
WATCH_PRUNE_EVERY = 2_000            # payments between sweeps of expired watchlist entries
TXN_FIELDS = ("ref", "ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_paid",
              "cur_paid", "cur_recv", "fmt")
ROUTING_KEEP = 500_000
MAX_REF_LEN = 128
WARM_QUEUE_BATCHES = 20_000


@dataclass
class Stats:
    batches: int = 0
    payments: int = 0
    rejected_not_owner: int = 0
    late: int = 0
    actions: dict = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    upstream_holds: int = 0
    escalations: int = 0
    watch_skipped_hub: int = 0
    engine_errors: int = 0
    stage_ms: dict = field(default_factory=dict)
    # security events: messages or payments refused at the boundary, by reason (crypto.Rejected)
    security: dict = field(default_factory=dict)

    def add_stage(self, name, ms, keep=50_000):
        self.stage_ms.setdefault(name, deque(maxlen=keep)).append(ms)

    def reject(self, reason: str, n: int = 1):
        self.security[reason] = self.security.get(reason, 0) + n

    def summary(self) -> dict:
        lat = {k: {f"p{int(q * 100)}": float(np.percentile(v, q * 100)) for q in (0.5, 0.95, 0.99)}
               for k, v in self.stage_ms.items() if v}
        return {"batches": self.batches, "payments": self.payments, "actions": dict(self.actions),
                "rejected_not_owner": self.rejected_not_owner, "late": self.late,
                "upstream_holds": self.upstream_holds, "escalations": self.escalations,
                "watch_skipped_hub": self.watch_skipped_hub, "engine_errors": self.engine_errors,
                "security": dict(self.security), "latency_ms": lat}


class Engine:
    def __init__(self, bundle: dict, bank_member: dict, pseudonym_key: bytes, usd_rates: dict,
                 t0, threads: int = 8, explain: bool = True):
        self._set_model(bundle)
        self._params = bundle["params"]
        self.graph = GraphFeatures(bundle["params"], threads=threads)
        self.groups = feature_groups(self.graph)
        self.explain = explain and self.scorer.can_explain
        self.bank_member = bank_member
        self.pseudo = Pseudonymizer(pseudonym_key, cache=False)
        self.rates = usd_rates
        self.t0 = pd.Timestamp(t0)
        self._t0_py = self.t0.to_pydatetime()
        self.last_t = -1
        self.watch: dict[int, int] = {}       # pseudonymous account -> watch expiry (data time)
        self.notices: dict[int, list] = {}    # member -> freeze requests for its own accounts
        # Receiving-account strings, kept in enclave memory only so a freeze request can name the
        # receiving bank's own account. Never written to features, logs or models.
        self.routing: OrderedDict[int, tuple] = OrderedDict()
        # (pseudonymous column dict, hot actions) for warm.py. Bounded: if no warm tier drains it
        # (warm tier off) or it falls far behind, the oldest batches are dropped, never the hot path.
        self.warm_queue: deque = deque(maxlen=WARM_QUEUE_BATCHES)
        self.warm_sink = None                 # set by WarmProcess: ship batches to the worker
        self.warm_poll = None                 # set by WarmProcess: apply returned escalations
        self.stats = Stats()
        self.lock = threading.RLock()        # escalate() may run inside _process
        self._next_id = 1 << 40               # edge ids for live payments
        self._since_prune = 0
        self.state = "priming"                # "ready" once prime() has loaded history
        self.started = time.time()

    def _set_model(self, bundle: dict):
        from .attest import bundle_digest
        self.scorer, self.policy = bundle["scorer"], bundle["policy"]
        self.hubs = np.asarray(bundle["hubs"], np.int64)
        self._hub_set = set(self.hubs.tolist())
        self.model_version = bundle.get("version", f"{bundle['tier']}-{self.scorer.kind}")
        self.model_digest = bundle_digest(bundle)

    def swap_model(self, bundle: dict):
        """Hot-swap a retrained model between micro-batches. The graph state is kept, so the new
        model must use the same GFP parameters; every decision carries the model version."""
        if bundle["params"] != self._params:
            raise ValueError("new model needs different graph features; restart the tier instead")
        with self.lock:
            self._set_model(bundle)
            self.explain = self.explain and self.scorer.can_explain

    def prime(self, df: pd.DataFrame):
        """Load pseudonymised, time-sorted history so feature windows are full at go-live."""
        with self.lock:
            self.graph.stream(edge_matrix(df, self.hubs))
            if len(df):
                self.last_t = max(self.last_t, int(df.t.iloc[-1]))
            self.state = "ready"

    def health(self) -> dict:
        return {"state": self.state, "model": self.model_version, "payments": self.stats.payments,
                "watchlist": len(self.watch), "uptime_s": round(time.time() - self.started, 1)}

    # ---- live path -----------------------------------------------------------------------
    def process(self, items: list[tuple[int, dict]]) -> tuple[dict, dict]:
        """Score (member, payment) pairs. Returns ({member: decisions}, {member: rejected refs})."""
        with self.lock:
            return self._process(items)

    def _frame(self, items):
        """Plain-Python batch preparation (pandas costs ~10 ms per call even for one payment).
        Returns column arrays, time-sorted; raw account strings do not survive this function
        except receiving accounts parked in self.routing for freeze notices."""
        rejected: dict[int, list] = {}
        rows = []
        bm, rates, t0 = self.bank_member, self.rates, self._t0_py
        not_owner = 0
        for m, p in items:
            try:
                ref = str(p["ref"])[:MAX_REF_LEN]
                fb, tb = str(p["from_bank"]), str(p["to_bank"])
                ts = datetime.fromisoformat(str(p["ts"]))
                amount = float(p["amount_paid"])
                if not (math.isfinite(amount) and amount >= 0) or not p["from_acct"] or not p["to_acct"]:
                    raise ValueError
                for k in ("cur_paid", "cur_recv", "fmt"):
                    p[k]                                       # noqa: B018 (must be present)
            except (KeyError, TypeError, ValueError, AttributeError):
                self.stats.reject("malformed")
                ref = str(p.get("ref", "?"))[:MAX_REF_LEN] if isinstance(p, dict) else "?"
                rejected.setdefault(int(m), []).append(ref)
                continue
            if bm.get(fb) != m:
                rejected.setdefault(int(m), []).append(ref)
                not_owner += 1
                continue
            rows.append((int((ts - t0).total_seconds()), m, {**p, "ref": ref}, fb, tb, ts))
        self.stats.rejected_not_owner += not_owner
        if not_owner:
            self.stats.reject("not_owner", not_owner)
        if not rows:
            return None, rejected
        rows.sort(key=lambda r: r[0])                      # stable: same-minute order kept
        n = len(rows)
        txn = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        t = np.fromiter((r[0] for r in rows), np.int64, n)
        late = t < self.last_t
        self.stats.late += int(late.sum())
        t[late] = self.last_t                              # late arrivals are graphed "now"
        pseudo = self.pseudo.one
        cols = {
            "txn_id": txn, "t": t,
            "member": [r[1] for r in rows],
            "ref": [r[2]["ref"] for r in rows],
            "ts": [r[5] for r in rows],
            "from_bank": [r[3] for r in rows], "to_bank": [r[4] for r in rows],
            "fmt": [str(r[2]["fmt"]) for r in rows],
            "cur_paid": [str(r[2]["cur_paid"]) for r in rows],
            "cur_recv": [str(r[2]["cur_recv"]) for r in rows],
            "amount_usd": np.fromiter((float(r[2]["amount_paid"]) * rates.get(str(r[2]["cur_paid"]), 1.0)
                                       for r in rows), np.float64, n),
            "src": np.fromiter((pseudo(r[3], r[2]["from_acct"]) for r in rows), np.int64, n),
            "dst": np.fromiter((pseudo(r[4], r[2]["to_acct"]) for r in rows), np.int64, n),
        }
        for tid, r in zip(txn.tolist(), rows):
            self.routing[tid] = (r[4], r[2]["to_acct"])
        while len(self.routing) > ROUTING_KEEP:
            self.routing.popitem(last=False)
        return cols, rejected

    def _process(self, items):
        t_start = time.perf_counter()
        if self.warm_poll is not None:
            self.warm_poll()
        c, rejected = self._frame(items)
        t_prep = time.perf_counter()
        if c is None:
            return {}, rejected
        n = len(c["txn_id"])
        edges = edge_arrays(c["txn_id"], c["src"], c["dst"], c["t"], c["amount_usd"], self.hubs)
        t = edges[:, 3]
        graph = np.empty((n, self.graph.n_features), np.float32)
        s = 0
        for e in np.append(np.flatnonzero(np.diff(t)) + 1, n):
            graph[s:e] = self.graph.update(edges[s:e])
            s = e
        self.last_t = max(self.last_t, int(t[-1]))
        raw = raw_features_live(c["amount_usd"], c["fmt"], c["cur_paid"], c["cur_recv"],
                                c["from_bank"], c["to_bank"], c["src"], c["dst"],
                                [x.hour for x in c["ts"]], [x.weekday() for x in c["ts"]], self.hubs)
        X = np.hstack([raw, graph])
        t_feat = time.perf_counter()
        p = self.policy.calibrate(self.scorer.predict(X, n_jobs=1))
        t_score = time.perf_counter()
        actions = self.policy.decide(p, c["amount_usd"])
        src, dst, tt = c["src"].tolist(), c["dst"].tolist(), c["t"].tolist()
        reason = [0] * n   # 0 model, 1 upstream flag
        watch, hubs = self.watch, self._hub_set
        for i in range(n):
            exp = watch.get(src[i])
            if exp is not None and exp >= tt[i] and actions[i] == RELEASE:
                actions[i] = HOLD
                reason[i] = 1
                self.stats.upstream_holds += 1
            if actions[i] != RELEASE:
                if dst[i] in hubs:
                    self.stats.watch_skipped_hub += 1
                else:
                    watch[dst[i]] = max(watch.get(dst[i], 0), tt[i] + WATCH_SECONDS)
        self._prune_watch(n)
        why: dict[int, list] = {}
        flagged = np.flatnonzero((actions != RELEASE) & (np.asarray(reason) == 0))
        if self.explain and len(flagged):
            contrib = self.scorer.contributions(X[flagged])
            if contrib is not None:
                for i, row in zip(flagged.tolist(), contrib):
                    why[i] = reason_codes(row, self.groups)
        t_explain = time.perf_counter()
        out: dict[int, list] = {}
        counts = self.stats.actions
        version = self.model_version
        for i, (ref, m, a, pi, r) in enumerate(zip(c["ref"], c["member"], actions.tolist(),
                                                   p.tolist(), reason)):
            counts[ACTIONS[a]] += 1
            d = {"ref": ref, "action": ACTIONS[a], "p": round(pi, 6),
                 "reason": "upstream_flag" if r else "model", "model": version}
            if r:
                d["why"] = [UPSTREAM]
            elif i in why:
                d["why"] = why[i]
            out.setdefault(int(m), []).append(d)
        for i in np.flatnonzero(actions == FREEZE):
            self._notify_receiver(int(c["txn_id"][i]), float(c["amount_usd"][i]),
                                  str(c["ts"][i]), "freeze_request")
        if self.warm_sink is not None:
            self.warm_sink(c, actions.copy())
        else:
            self.warm_queue.append((c, actions.copy()))
        t_end = time.perf_counter()
        st = self.stats
        st.batches += 1
        st.payments += n
        st.add_stage("prepare", (t_prep - t_start) * 1e3)
        st.add_stage("features", (t_feat - t_prep) * 1e3)
        st.add_stage("score", (t_score - t_feat) * 1e3)
        if len(flagged) and self.explain:
            st.add_stage("explain", (t_explain - t_score) * 1e3)
        st.add_stage("decide", (t_end - t_score) * 1e3)
        st.add_stage("engine_total", (t_end - t_start) * 1e3)
        return out, rejected

    def _prune_watch(self, n: int):
        """Drop expired watchlist entries every WATCH_PRUNE_EVERY payments (bounded memory)."""
        self._since_prune += n
        if self._since_prune < WATCH_PRUNE_EVERY:
            return
        self._since_prune = 0
        now = self.last_t
        for k in [k for k, exp in self.watch.items() if exp < now]:
            del self.watch[k]

    def _notify_receiver(self, txn_id: int, amount_usd: float, ts: str, kind: str):
        """Tell the receiving bank which of *its own* accounts to act on. No sender details."""
        bank, acct = self.routing.get(txn_id, (None, None))
        m = self.bank_member.get(bank)
        if m is None:
            return
        self.notices.setdefault(int(m), []).append({
            "case": secrets.token_hex(6), "kind": kind, "to_bank": bank, "to_acct": acct,
            "amount_usd": round(amount_usd, 2), "ts": ts})

    def escalate(self, txn_id: int, dst_pid: int, t: int, amount_usd: float, ts: str):
        """Warm tier found a released payment suspicious: watch the receiver, notify its bank."""
        with self.lock:
            if int(dst_pid) in self._hub_set:
                self.stats.watch_skipped_hub += 1
            else:
                self.watch[dst_pid] = max(self.watch.get(dst_pid, 0), t + WATCH_SECONDS)
            self.stats.escalations += 1
            self._notify_receiver(txn_id, amount_usd, ts, "escalation_hold")

    def drain_notices(self, member: int) -> list:
        with self.lock:
            return self.notices.pop(member, [])
