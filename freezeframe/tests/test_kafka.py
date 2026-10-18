"""Kafka transport end to end. Needs a broker: FF_KAFKA=localhost:9092 pytest tests/test_kafka.py"""
import asyncio
import copy
import os
import uuid

import pytest

from freezeframe.consortium import assign_members
from freezeframe.demo_setup import to_payment
from freezeframe.enclave.crypto import KeyPair
from freezeframe.enclave.engine import Engine
from freezeframe.enclave.stream import BankClient, HotConsumer, KafkaBus, decisions_topic

from .conftest import KEY

BROKER = os.environ.get("FF_KAFKA")
pytestmark = pytest.mark.skipif(not BROKER, reason="set FF_KAFKA=host:port to run")


def test_kafka_roundtrip(raw_df, pseudo_df, bundles):
    bm = assign_members(raw_df, 0)
    eng = Engine(copy.deepcopy(bundles["hot"]), bm, KEY, {"US Dollar": 1.0}, t0=raw_df.ts.min(),
                 threads=2)
    eng.prime(pseudo_df[pseudo_df.split < 2])
    keys = KeyPair()
    run = uuid.uuid4().hex[:8]

    async def go():
        bus = KafkaBus(BROKER, group=f"ff-test-{run}", prefix=f"t{run}.")
        banks = {m: BankClient(m, bus, keys.identity) for m in range(5)}
        cons = HotConsumer(eng, bus, keys, {m: b.identity for m, b in banks.items()})
        task = asyncio.create_task(cons.run())
        live = raw_df[raw_df.split == 2].head(60)
        for r in live.itertuples():
            await banks[bm[str(r.from_bank)]].send(to_payment(r))
        for _ in range(600):
            if cons.decided >= len(live):
                break
            await asyncio.sleep(0.05)
        cons.stop()
        await task
        got = {}
        for m, b in banks.items():
            if (live.from_bank.astype(str).map(bm) == m).any():
                while len([v for v in got.values() if v == m]) < (live.from_bank.astype(str).map(bm) == m).sum():
                    msg = await bus.get(decisions_topic(m), timeout=20)
                    for d in b.open(msg)["decisions"]:
                        got[d["ref"]] = m
        await bus.close()
        return live, got, cons

    live, got, cons = asyncio.run(go())
    assert set(got) == set(live.txn_id.astype(str))
    assert all(got[str(r.txn_id)] == bm[str(r.from_bank)] for r in live.itertuples())
