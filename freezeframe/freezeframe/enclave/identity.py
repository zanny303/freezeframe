"""Who the consortium members are, and how they join.

Each member registers two public keys with the enclave: an X25519 box key (decisions are
sealed to it) and an Ed25519 verify key (its submissions must be signed by the matching key).

Enrollment is the only unauthenticated step, so it is gated by a one-time enrollment token per
member, delivered out of band by the consortium operator. A token works once; after that the
member's keys can only be replaced by a request signed with the member's *current* key
(rotation). Before this, anyone could overwrite member m's key over HTTP and read m's decisions.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path

from .crypto import Identity, Rejected


def _h(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class Registry:
    def __init__(self, tokens: dict[int, str] | None = None):
        self._token_hash = {int(m): _h(t) for m, t in (tokens or {}).items()}
        self._members: dict[int, Identity] = {}
        self._lock = threading.Lock()

    # ---- operator side -------------------------------------------------------------------
    @classmethod
    def with_tokens(cls, members, path: Path | None = None) -> tuple["Registry", dict[int, str]]:
        """New random one-time enrollment tokens; optionally written (0600) for the operator."""
        tokens = {int(m): secrets.token_urlsafe(24) for m in members}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({str(m): t for m, t in tokens.items()}, f, indent=1)
        return cls(tokens), tokens

    def set(self, m: int, ident: Identity):
        """Direct registration, for deployments where the operator wires the banks in-process
        (demo, benchmark, tests)."""
        with self._lock:
            self._members[int(m)] = ident

    # ---- member side ---------------------------------------------------------------------
    def enroll(self, m: int, ident: Identity, token: str):
        m = int(m)
        with self._lock:
            expected = self._token_hash.get(m)
            if expected is None or not hmac.compare_digest(expected, _h(str(token))):
                raise Rejected("unknown_sender")
            del self._token_hash[m]          # one-time
            self._members[m] = ident

    def rotate(self, m: int, ident: Identity):
        """Caller must already have verified a request signed with m's current key."""
        with self._lock:
            if int(m) not in self._members:
                raise Rejected("unknown_sender")
            self._members[int(m)] = ident

    # ---- lookups -------------------------------------------------------------------------
    def get(self, m: int) -> Identity | None:
        return self._members.get(int(m))

    def members(self) -> list[int]:
        return sorted(self._members)

    def verify_key_for(self, sender: str) -> str | None:
        if not isinstance(sender, str) or not sender.startswith("member:"):
            return None
        try:
            ident = self._members.get(int(sender.split(":", 1)[1]))
        except ValueError:
            return None
        return ident.verify_b64 if ident else None

    def __contains__(self, m) -> bool:
        return int(m) in self._members

    @classmethod
    def of(cls, members: "Registry | dict[int, Identity]") -> "Registry":
        if isinstance(members, Registry):
            return members
        reg = cls()
        for m, ident in members.items():
            reg.set(m, ident)
        return reg
