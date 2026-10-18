"""Streaming hot path: banks publish sealed payments, the enclave consumes micro-batches and
publishes sealed decisions on one topic per member.

Topics
  payments.in        {"member", "sealed"}   signed by member m, sealed to the enclave; the body
                     is {"payments": [...], "sent_at": [...]}: a bank gateway batches for a few ms
  decisions.<m>      {"sealed"}             signed by the enclave, sealed to member m
  notices.<m>        {"sealed"}             freeze requests / escalations for m's own accounts

Every message is sign-then-seal with replay protection (crypto.py). The cleartext ``member``
field is only a routing hint: the enclave accepts a message only if it verifies under the key
registered for that member, so a bank cannot submit as another bank, and an attacker on the bus
cannot inject, replay or tamper with payments, or forge a decision on a bank's topic.

``InProcessBus`` is the default transport (asyncio queues). ``KafkaBus`` speaks the same
interface over Apache Kafka / IBM Event Streams.
"""
import asyncio
import json
import logging
import time
from collections import deque

from .crypto import (ENCLAVE, Identity, KeyPair, Rejected, ReplayGuard, member_id, open_signed,
                     seal_signed)
from .identity import Registry

log = logging.getLogger("freezeframe.stream")

PAYMENTS = "payments.in"


def decisions_topic(m: int) -> str:
    return f"decisions.{m}"


def notices_topic(m: int) -> str:
    return f"notices.{m}"


class InProcessBus:
    """``tap``, if set, sees every published message (the demo's "attacker on the wire")."""

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.tap = None

    def _q(self, topic):
        return self.queues.setdefault(topic, asyncio.Queue())

    async def publish(self, topic: str, msg: dict):
        if self.tap is not None:
            self.tap(topic, msg)
        self._q(topic).put_nowait(msg)

    async def get(self, topic: str, timeout: float | None = None):
        if timeout is None:
            return await self._q(topic).get()
        return await asyncio.wait_for(self._q(topic).get(), timeout)

    def get_nowait(self, topic: str):
        try:
            return self._q(topic).get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def close(self):
        pass


class KafkaBus:
    """Same interface over Kafka (kafka-python). One partition on payments.in keeps time order."""

    def __init__(self, bootstrap: str = "localhost:9092", group: str = "freezeframe-enclave",
                 prefix: str = ""):
        from kafka import KafkaConsumer, KafkaProducer
        self.prefix = prefix
        self._producer = KafkaProducer(bootstrap_servers=bootstrap, linger_ms=1,
                                       value_serializer=lambda v: json.dumps(v).encode())
        self._consumers = {}
        self._mk = lambda topic: KafkaConsumer(
            prefix + topic, bootstrap_servers=bootstrap, group_id=f"{group}-{topic}",
            auto_offset_reset="earliest", enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b))
        self._buf: dict[str, list] = {}

    def _consumer(self, topic):
        if topic not in self._consumers:
            self._consumers[topic] = self._mk(topic)
        return self._consumers[topic]

    async def publish(self, topic, msg):
        self._producer.send(self.prefix + topic, msg)

    def _poll(self, topic, timeout_ms):
        buf = self._buf.setdefault(topic, [])
        if not buf:
            for recs in self._consumer(topic).poll(timeout_ms=timeout_ms, max_records=5000).values():
                buf.extend(r.value for r in recs)
        return buf.pop(0) if buf else None

    async def get(self, topic, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._producer.flush()
            msg = await asyncio.to_thread(self._poll, topic, 50)
            if msg is not None:
                return msg
            if deadline and time.monotonic() > deadline:
                raise asyncio.TimeoutError

    def get_nowait(self, topic):
        buf = self._buf.get(topic)
        return buf.pop(0) if buf else None

    async def close(self):
        self._producer.flush()
        self._producer.close()
        for c in self._consumers.values():
            c.close()


class BankClient:
    """A consortium member: signs and seals its own payments to the enclave, reads only its own
    topics, and accepts only messages signed by the enclave it attested.

    With ``linger_ms`` > 0 the client batches payments for that long before sealing (one
    public-key operation per batch instead of per payment); latency is still measured from each
    payment's own send time, so the linger is counted against the budget.
    """

    def __init__(self, member: int, bus, enclave: Identity, linger_ms: float = 0.0):
        if not isinstance(enclave, Identity):
            raise TypeError("enclave must be an Identity (use BankClient.from_attestation)")
        self.member = member
        self.me = member_id(member)
        self.keys = KeyPair()
        self.bus = bus
        self.enclave = enclave
        self.linger = linger_ms / 1e3
        self._buf: list = []
        self._first = 0.0
        self.guard = ReplayGuard()
        self.rejected: dict[str, int] = {}     # inbound messages refused, by reason
        self.last_sealed: dict | None = None   # last message this bank published (demo display)

    @classmethod
    def from_attestation(cls, member: int, bus, doc: dict, expected_measurement: str | None = None,
                         expected_models: dict | None = None, allow_simulated: bool = True,
                         linger_ms: float = 0.0) -> "BankClient":
        """Verify the enclave's attestation document before trusting its keys."""
        from .attest import verify_attestation
        ident = verify_attestation(doc, expected_measurement, expected_models, allow_simulated)
        return cls(member, bus, ident, linger_ms=linger_ms)

    @property
    def identity(self) -> Identity:
        return self.keys.identity

    async def send(self, payment: dict, sent_at: float | None = None):
        sent_at = time.time() if sent_at is None else sent_at
        if self.linger <= 0:
            await self._publish([payment], [sent_at])
            return
        if not self._buf:
            self._first = time.monotonic()
        self._buf.append((payment, sent_at))
        if time.monotonic() - self._first >= self.linger:
            await self.flush()

    async def flush(self):
        if self._buf:
            buf, self._buf = self._buf, []
            await self._publish([p for p, _ in buf], [t for _, t in buf])

    async def run_flusher(self, stop: asyncio.Event):
        while not stop.is_set():
            await asyncio.sleep(self.linger or 0.005)
            if self._buf and time.monotonic() - self._first >= self.linger:
                await self.flush()
        await self.flush()

    def seal(self, body) -> dict:
        return {"member": self.member, "sealed": seal_signed(body, self.me, self.keys, ENCLAVE,
                                                             self.enclave.box_b64)}

    async def _publish(self, payments, sent_at):
        msg = self.seal({"payments": payments, "sent_at": sent_at})
        self.last_sealed = msg
        await self.bus.publish(PAYMENTS, msg)

    def open(self, msg: dict):
        """Body of a message from the enclave; raises Rejected for anything it did not sign."""
        try:
            return open_signed(msg["sealed"], self.keys, self.me, self.enclave.verify_b64,
                               self.guard, expect_from=ENCLAVE)[1]
        except Rejected as e:
            self.rejected[e.reason] = self.rejected.get(e.reason, 0) + 1
            raise
        except (KeyError, TypeError):
            self.rejected["malformed"] = self.rejected.get("malformed", 0) + 1
            raise Rejected("malformed") from None


class HotConsumer:
    """Drains payments.in into micro-batches and runs them through the engine.

    A message that fails any check (cannot be opened, bad signature, wrong member, stale,
    replayed, malformed) is dropped and counted in ``engine.stats.security``; it never stops
    the consumer. ``on_reject(reason, msg)`` lets an operator console show it live.
    """

    LATENCY_KEEP = 1_000_000

    def __init__(self, engine, bus, enclave_keys: KeyPair, members: "Registry | dict[int, Identity]",
                 linger_ms: float = 2.0, max_batch: int = 500, on_reject=None):
        self.engine, self.bus, self.keys = engine, bus, enclave_keys
        self.registry = Registry.of(members)
        self.guard = ReplayGuard()
        self.linger = linger_ms / 1e3
        self.max_batch = max_batch
        self.e2e_ms: deque = deque(maxlen=self.LATENCY_KEEP)
        self.batch_sizes: deque = deque(maxlen=100_000)
        self.decided = 0                      # payments decided (e2e_ms keeps only the latest)
        self.on_reject = on_reject
        self._stop = asyncio.Event()

    def stop(self):
        self._stop.set()

    async def _collect(self):
        try:
            first = await self.bus.get(PAYMENTS, timeout=0.2)
        except asyncio.TimeoutError:
            return []
        batch = [first]
        deadline = time.monotonic() + self.linger
        while len(batch) < self.max_batch:   # messages; each may carry several payments
            msg = self.bus.get_nowait(PAYMENTS)
            if msg is None:
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0)
                continue
            batch.append(msg)
        return batch

    def _reject(self, reason: str, msg):
        self.engine.stats.reject(reason)
        if self.on_reject is not None:
            try:
                self.on_reject(reason, msg)
            except Exception:                 # an observer must never stop the enclave
                log.exception("on_reject observer failed")

    def open(self, msg) -> tuple[int, list, list]:
        """(member, payments, sent_at) for an authentic message; raises Rejected otherwise."""
        try:
            m = int(msg["member"])
            token = msg["sealed"]
        except (KeyError, TypeError, ValueError):
            raise Rejected("malformed") from None
        _, body = open_signed(token, self.keys, ENCLAVE, self.registry.verify_key_for, self.guard,
                              expect_from=member_id(m))
        try:
            payments, sent = list(body["payments"]), [float(x) for x in body["sent_at"]]
        except (KeyError, TypeError, ValueError):
            raise Rejected("malformed") from None
        if len(payments) != len(sent) or not all(isinstance(p, dict) for p in payments):
            raise Rejected("malformed")
        return m, payments, sent

    def seal_for(self, member: int, body) -> dict:
        ident = self.registry.get(member)
        return {"sealed": seal_signed(body, ENCLAVE, self.keys, member_id(member), ident.box_b64)}

    async def run(self):
        while not self._stop.is_set():
            batch = await self._collect()
            if not batch:
                continue
            t0 = time.perf_counter()
            items, sent = [], []
            for msg in batch:
                try:
                    m, payments, sent_at = self.open(msg)
                except Rejected as e:
                    self._reject(e.reason, msg)
                    continue
                items.extend((m, p) for p in payments)
                sent.extend(sent_at)
            self.engine.stats.add_stage("decrypt", (time.perf_counter() - t0) * 1e3)
            if not items:
                continue
            try:
                out, rejected = self.engine.process(items)
            except Exception:
                log.exception("engine failed on a micro-batch of %d payments", len(items))
                self.engine.stats.engine_errors += 1
                continue
            t_pub = time.perf_counter()
            for member in set(out) | set(rejected):
                if member not in self.registry:
                    continue
                await self.bus.publish(decisions_topic(member), self.seal_for(member, {
                    "decisions": out.get(member, []), "rejected": rejected.get(member, [])}))
            for member in self.registry.members():   # receivers may have no payments here
                notes = self.engine.drain_notices(member)
                if notes:
                    await self.bus.publish(notices_topic(member), self.seal_for(member, notes))
            now = time.time()
            self.engine.stats.add_stage("publish", (time.perf_counter() - t_pub) * 1e3)
            self.e2e_ms.extend((now - ts) * 1e3 for ts in sent)
            self.decided += len(sent)
            self.batch_sizes.append(len(items))
            await asyncio.sleep(0)
