"""What a bank checks before it seals a single payment to the enclave.

The enclave publishes an *attestation document*: its two public keys, a measurement of the code
it runs, a digest of the models it scores with, and whether it is running as an IBM Secure
Execution guest. Banks pin the measurement they expect (from a reproducible build of this repo)
and refuse to talk to an enclave that reports anything else.

Honesty note: outside Secure Execution the document is only *self-signed* by the enclave's own
Ed25519 key, so it proves "whoever holds this key claims this measurement", not that hardware
measured it. ``binding`` says which case applies, and ``verify_attestation(...,
allow_simulated=False)`` is the production policy. Under IBM Confidential Computing Container
Runtime (Hyper Protect), the hardware-signed attestation record carries the image measurement
and the enclave puts a hash of its public keys into it (Roadmap: Secure Execution).
"""
import hashlib
import pickle
import platform
import time
from pathlib import Path

from nacl.encoding import Base64Encoder
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from .crypto import Identity, KeyPair, Rejected, canonical

PACKAGE_DIR = Path(__file__).resolve().parent.parent      # freezeframe/
MAX_DOC_AGE_S = 24 * 3600


def attestation_status() -> str:
    """'secure-execution' inside an IBM Secure Execution guest, otherwise 'simulated'."""
    p = Path("/sys/firmware/uv/prot_virt_guest")
    try:
        return "secure-execution" if p.read_text().strip() == "1" else "simulated"
    except OSError:
        return "simulated"


def code_measurement(root: Path = PACKAGE_DIR) -> str:
    """SHA-256 over every enclave source file (path + content), in a fixed order."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(f.relative_to(root).as_posix().encode() + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def bundle_digest(bundle: dict) -> str:
    """Digest of a model bundle: the file bytes when known (load_bundle records them), else the
    pickled bundle (an XGBoost scorer pickles as its portable JSON)."""
    if bundle.get("digest"):
        return bundle["digest"]
    return hashlib.sha256(pickle.dumps(bundle)).hexdigest()


def keys_hash(ident: Identity) -> str:
    """What goes into a hardware attestation record's user data to bind the keys to it."""
    return hashlib.sha256(f"{ident.box_b64}|{ident.verify_b64}".encode()).hexdigest()


def attestation_doc(keys: KeyPair, models: dict[str, str], measurement: str | None = None) -> dict:
    status = attestation_status()
    body = {
        "enclave_box_key": keys.public_b64,
        "enclave_verify_key": keys.verify_b64,
        "keys_hash": keys_hash(keys.identity),
        "code_measurement": measurement or code_measurement(),
        "models": dict(models),
        "status": status,
        "binding": "secure-execution" if status == "secure-execution" else "self-signed (simulated)",
        "machine": platform.machine(),
        "issued_at": time.time(),
    }
    return {**body, "signature": keys.sign_b64(canonical(body))}


def verify_attestation(doc: dict, expected_measurement: str | None = None,
                       expected_models: dict[str, str] | None = None,
                       allow_simulated: bool = True, now: float | None = None) -> Identity:
    """Bank-side check. Returns the enclave identity to seal to, or raises Rejected."""
    try:
        body = {k: v for k, v in doc.items() if k != "signature"}
        vk = VerifyKey(body["enclave_verify_key"].encode(), encoder=Base64Encoder)
        vk.verify(canonical(body), Base64Encoder.decode(doc["signature"].encode()))
        ident = Identity(body["enclave_box_key"], body["enclave_verify_key"])
    except (KeyError, TypeError, ValueError, BadSignatureError):
        raise Rejected("bad_signature") from None
    if body.get("keys_hash") != keys_hash(ident):
        raise Rejected("bad_signature")
    if expected_measurement is not None and body.get("code_measurement") != expected_measurement:
        raise Rejected("wrong_recipient")
    for tier, digest in (expected_models or {}).items():
        if body.get("models", {}).get(tier) != digest:
            raise Rejected("wrong_recipient")
    if not allow_simulated and body.get("status") != "secure-execution":
        raise Rejected("wrong_recipient")
    now = time.time() if now is None else now
    if abs(now - float(body.get("issued_at", 0))) > MAX_DOC_AGE_S:
        raise Rejected("stale")
    return ident
