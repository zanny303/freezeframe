"""HTTP face of the enclave (request/response alternative to the stream).

  GET  /v1/attestation          signed attestation document: keys, code measurement, model
                                digests, Secure Execution status. Banks verify it first.
  GET  /v1/enclave              short form (public keys, attestation status, model)
  GET  /v1/health               "priming" while the graph loads history, then "ready"
  POST /v1/members/{m}/enroll   {"box_public_key", "verify_key", "token"}: one-time enrollment
                                token from the operator; works once
  POST /v1/members/{m}/rotate   {"sealed"}: new keys, signed with m's *current* key
  POST /v1/submit               {"member", "sealed"} signed + sealed payment batch
                                -> {"sealed"} decisions for that member only
  POST /v1/notices              {"member", "sealed"} signed request
                                -> {"sealed"} freeze requests for m's own accounts
  GET  /v1/metrics              consortium-level counts, latency and security events;
                                no transaction data

Every request body except enrollment is signed by the member and sealed to the enclave, with
replay protection; every response is signed by the enclave and sealed to the member
(crypto.py). Failed checks return 401/403 and are counted under ``security`` in /v1/metrics.

Run:  python -m freezeframe.enclave.service     (enrollment tokens are written, 0600, to
      artifacts/enrollment_tokens.json for the operator to hand to each bank)
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import config
from .attest import attestation_doc, attestation_status, code_measurement  # noqa: F401 (re-export)
from .crypto import ENCLAVE, Identity, KeyPair, Rejected, ReplayGuard, member_id, open_signed, seal_signed
from .identity import Registry


class Submission(BaseModel):
    member: int
    sealed: str


class Enrollment(BaseModel):
    box_public_key: str
    verify_key: str
    token: str


class Sealed(BaseModel):
    sealed: str


_STATUS = {"unknown_sender": 403, "sender_mismatch": 403, "not_owner": 403}


def create_app(engine, keys: KeyPair, warm=None, registry: Registry | None = None) -> FastAPI:
    """``registry`` defaults to an empty one whose enrollment tokens are in
    ``app.state.enrollment_tokens`` (tests and in-process operators read them there)."""
    app = FastAPI(title="FreezeFrame enclave")
    if registry is None:
        registry, tokens = Registry.with_tokens(range(config.N_MEMBERS))
        app.state.enrollment_tokens = tokens
    guard = ReplayGuard()
    measurement = code_measurement()
    app.state.engine, app.state.registry = engine, registry

    def models() -> dict:
        out = {"hot": engine.model_digest}
        if warm is not None and getattr(warm, "model_digest", None):
            out["full"] = warm.model_digest
        return out

    def refuse(reason: str):
        engine.stats.reject(reason)
        raise HTTPException(_STATUS.get(reason, 401), reason)

    def open_from(member: int, token: str):
        if member not in registry:
            refuse("unknown_sender")
        try:
            return open_signed(token, keys, ENCLAVE, registry.verify_key_for, guard,
                               expect_from=member_id(member))[1]
        except Rejected as e:
            refuse(e.reason)

    def reply(member: int, body) -> dict:
        return {"sealed": seal_signed(body, ENCLAVE, keys, member_id(member),
                                      registry.get(member).box_b64)}

    @app.get("/v1/attestation")
    def attestation():
        return attestation_doc(keys, models(), measurement)

    @app.get("/v1/enclave")
    def enclave():
        return {"public_key": keys.public_b64, "verify_key": keys.verify_b64,
                "attestation": attestation_status(), "model": engine.model_version,
                "code_measurement": measurement}

    @app.get("/v1/health")
    def health():
        out = engine.health()
        if out["state"] != "ready":
            raise HTTPException(503, out)
        return out

    @app.post("/v1/members/{m}/enroll")
    def enroll(m: int, body: Enrollment):
        try:
            registry.enroll(m, Identity(body.box_public_key, body.verify_key), body.token)
        except Rejected as e:
            refuse(e.reason)
        return {"ok": True, "member": m}

    @app.post("/v1/members/{m}/rotate")
    def rotate(m: int, body: Sealed):
        new = open_from(m, body.sealed)
        try:
            registry.rotate(m, Identity(str(new["box_public_key"]), str(new["verify_key"])))
        except (KeyError, TypeError):
            refuse("malformed")
        return {"ok": True, "member": m}

    @app.post("/v1/submit")
    def submit(body: Submission):
        batch = open_from(body.member, body.sealed)
        payments = batch.get("payments") if isinstance(batch, dict) else None
        if not isinstance(payments, list) or not all(isinstance(p, dict) for p in payments):
            refuse("malformed")
        out, rejected = engine.process([(body.member, p) for p in payments])
        return reply(body.member, {"decisions": out.get(body.member, []),
                                   "rejected": rejected.get(body.member, [])})

    @app.post("/v1/notices")
    def notices(body: Submission):
        req = open_from(body.member, body.sealed)
        if not isinstance(req, dict) or req.get("op") != "notices":
            refuse("malformed")
        return reply(body.member, engine.drain_notices(body.member))

    @app.get("/v1/metrics")
    def metrics():
        out = engine.stats.summary()
        if warm is not None:
            out["warm"] = warm.summary()
        out["replay_cache"] = len(guard)
        return out

    return app


def main():
    import uvicorn
    from ..demo_setup import build_engine
    engine, keys, warm, _ = build_engine(os.environ.get("FF_DATASET", "HI-Small"))
    warm.start()
    path = config.ARTIFACTS_DIR / "enrollment_tokens.json"
    registry, _ = Registry.with_tokens(range(config.N_MEMBERS), path)
    print(f"enrollment tokens for the {config.N_MEMBERS} members: {path} (hand out of band)")
    uvicorn.run(create_app(engine, keys, warm, registry), host="0.0.0.0",
                port=int(os.environ.get("PORT", 8443)))


if __name__ == "__main__":
    main()
