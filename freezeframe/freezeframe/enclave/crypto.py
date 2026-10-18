"""Transport security between banks and the enclave (libsodium via PyNaCl).

Every message, in both directions, is **sign-then-seal**:

1. The sender builds an envelope ``{"v", "from", "to", "nonce", "issued_at", "body"}`` and signs
   it with its Ed25519 key. ``from``/``to`` are identities such as ``"member:3"`` or
   ``"enclave"``.
2. It seals the signed bytes to the recipient's X25519 public key (``SealedBox``).

The recipient opens the box, checks the signature against the key *registered* for the claimed
sender, checks that the envelope is addressed to itself, rejects envelopes outside a freshness
window, and rejects a ``(sender, nonce)`` pair it has already seen. Together this gives:

- confidentiality: only the recipient can read the body (sealed box);
- authenticity: a bank cannot submit payments as another member, and nobody can forge an
  enclave decision on a bank's topic (Ed25519);
- no surreptitious forwarding: an envelope opened by one party cannot be re-sealed to another
  and still verify, because the signed envelope names its recipient;
- replay protection: a captured message is rejected the second time (nonce + freshness window).

Failures raise :class:`Rejected` with a short machine-readable reason, which the enclave counts
as a security event instead of crashing.

In production the enclave's keys are bound to an IBM Secure Execution attestation record
(see ``attest.py``), so a bank only seals payments to a measured enclave image.
"""
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from nacl.encoding import Base64Encoder
from nacl.exceptions import BadSignatureError, CryptoError
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.signing import SigningKey, VerifyKey

ENCLAVE = "enclave"
MAX_SKEW_S = 120.0
SIGNATURE_BYTES = 64

REJECT_REASONS = ("undecryptable", "malformed", "unknown_sender", "sender_mismatch",
                  "wrong_recipient", "bad_signature", "stale", "replay", "not_owner")


def member_id(m: int) -> str:
    return f"member:{int(m)}"


class Rejected(Exception):
    """A message failed a security check. ``reason`` is one of REJECT_REASONS."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Identity:
    """The public half of a party: where to seal to, and which key its signatures verify with."""
    box_b64: str
    verify_b64: str


class KeyPair:
    """X25519 key for sealed boxes (confidentiality) + Ed25519 key for signatures (authenticity)."""

    def __init__(self, sk: PrivateKey | None = None, signing: SigningKey | None = None):
        self.sk = sk or PrivateKey.generate()
        self.signing = signing or SigningKey.generate()

    @property
    def public_b64(self) -> str:
        return self.sk.public_key.encode(Base64Encoder).decode()

    @property
    def verify_b64(self) -> str:
        return self.signing.verify_key.encode(Base64Encoder).decode()

    @property
    def identity(self) -> Identity:
        return Identity(self.public_b64, self.verify_b64)

    def sign_b64(self, data: bytes) -> str:
        return Base64Encoder.encode(self.signing.sign(data).signature).decode()


class ReplayGuard:
    """Remembers ``(sender, nonce)`` pairs long enough that any replay is caught either here or
    by the freshness check (an entry inserted at T for a message issued at I, |T - I| <= skew,
    is kept until T + 2 * skew; a replay after that is older than the skew window)."""

    def __init__(self, skew_s: float = MAX_SKEW_S):
        self.skew = skew_s
        self._seen: set = set()
        self._order: deque = deque()
        self._lock = threading.Lock()

    def check(self, sender: str, nonce: str, now: float) -> bool:
        key = (sender, nonce)
        with self._lock:
            while self._order and self._order[0][0] < now:
                self._seen.discard(self._order.popleft()[1])
            if key in self._seen:
                return False
            self._seen.add(key)
            self._order.append((now + 2 * self.skew, key))
            return True

    def __len__(self):
        return len(self._seen)


def canonical(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str).encode()


def seal_signed(body, sender: str, keys: KeyPair, recipient: str, recipient_box_b64: str,
                issued_at: float | None = None) -> str:
    """Sign an envelope as ``sender`` and seal it to ``recipient``'s box key."""
    env = {"v": 1, "from": sender, "to": recipient, "nonce": os.urandom(12).hex(),
           "issued_at": time.time() if issued_at is None else issued_at, "body": body}
    signed = keys.signing.sign(json.dumps(env, separators=(",", ":"), default=str).encode())
    box = SealedBox(PublicKey(recipient_box_b64.encode(), encoder=Base64Encoder))
    return box.encrypt(bytes(signed), encoder=Base64Encoder).decode()


def open_signed(token: str, keys: KeyPair, me: str, verify_key_for, guard: ReplayGuard | None,
                expect_from: str | None = None, skew_s: float = MAX_SKEW_S,
                now: float | None = None) -> tuple[str, object]:
    """Open, authenticate and de-duplicate a message addressed to ``me``. Returns (sender, body).

    ``verify_key_for`` is a base64 verify key, or a callable sender -> verify key (or None if the
    sender is not registered).
    """
    now = time.time() if now is None else now
    try:
        signed = SealedBox(keys.sk).decrypt(str(token).encode(), encoder=Base64Encoder)
    except (CryptoError, ValueError, TypeError):
        raise Rejected("undecryptable") from None
    try:
        env = json.loads(signed[SIGNATURE_BYTES:])
        sender, recipient, nonce = env["from"], env["to"], str(env["nonce"])
        issued = float(env["issued_at"])
        body = env["body"]
    except (ValueError, KeyError, TypeError):
        raise Rejected("malformed") from None
    if expect_from is not None and sender != expect_from:
        raise Rejected("sender_mismatch")
    vk = verify_key_for(sender) if callable(verify_key_for) else verify_key_for
    if not vk:
        raise Rejected("unknown_sender")
    try:
        VerifyKey(vk.encode(), encoder=Base64Encoder).verify(signed)
    except (BadSignatureError, ValueError, TypeError):
        raise Rejected("bad_signature") from None
    if recipient != me:
        raise Rejected("wrong_recipient")
    if abs(now - issued) > skew_s:
        raise Rejected("stale")
    if guard is not None and not guard.check(sender, nonce, now):
        raise Rejected("replay")
    return sender, body


# ---- unauthenticated sealing (kept for tooling; the enclave never accepts it) ------------------

def seal(obj, public_b64: str) -> str:
    pk = PublicKey(public_b64.encode(), encoder=Base64Encoder)
    return SealedBox(pk).encrypt(json.dumps(obj, default=str).encode(), encoder=Base64Encoder).decode()


def open_unsigned(keys: KeyPair, token: str):
    return json.loads(SealedBox(keys.sk).decrypt(token.encode(), encoder=Base64Encoder))
