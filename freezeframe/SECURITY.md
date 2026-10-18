# FreezeFrame security model

## Parties and what each may learn

| Party | Trusted for | Must not learn |
|---|---|---|
| Member bank | its own customers' data | other members' payments, accounts, features or decisions |
| Enclave (IBM Secure Execution guest) | running the measured code | nothing leaves it except per-owner decisions, receiver-only freeze notices and consortium counts |
| Host operator / cloud admin | uptime | anything inside the enclave (Secure Execution keeps it out; **simulated** in this build) |
| Network, bus (Kafka / Event Streams), other tenants | delivery only | message contents; must not be able to inject, replay or alter traffic |

## Mechanisms

1. **Sign-then-seal** (`freezeframe/enclave/crypto.py`). The envelope `{from, to, nonce, issued_at, body}` is Ed25519-signed by the sender and sealed (X25519 sealed box) to the recipient. The recipient checks the signature against the *registered* key for the claimed sender, checks `to` is itself (no forwarding), a 120 s freshness window, and a `(sender, nonce)` replay cache. This applies to bank → enclave and enclave → bank.
2. **Enrollment** (`identity.py`). One-time token per member, delivered out of band by the consortium operator; the token is spent on first use. Key rotation needs a request signed by the current key.
3. **Attestation** (`attest.py`). The enclave publishes its keys, a SHA-256 measurement of its code, the digests of its model files and its Secure Execution status, signed by its key. Banks pin the expected measurement (`BankClient.from_attestation`). Production policy is `allow_simulated=False`, with the keys' hash carried in the hardware attestation record (see the Roadmap in README.md).
4. **Pseudonymisation** (`pseudonym.py`). HMAC-SHA256 with a key only the enclave holds, before any processing. Tests scan features, queued frames, metrics and model files for raw account strings.
5. **Ownership and routing** (`engine.py`). A payment is accepted only from the member that owns its sending bank; decisions go only to that member; freeze notices go only to the receiving bank and name only its own account.
6. **Untrusted input** (`stream.HotConsumer`, `Engine._frame`). Malformed messages and payments are refused and counted; they never stop the consumer.
7. **Reason codes** (`explain.py`). Only evidence *families* with their share of the score, never counterparties, banks, counts or amounts.

All refusals are counted under `security` in `/v1/metrics` and shown on the dashboard.

## Production hardening

- **No TLS in the HTTP service.** Payloads are sealed and signed end to end, but production should still run behind TLS/mTLS and per-member Kafka ACLs (see the Roadmap in README.md).
- **Model files are pickles.** Only load bundles the consortium trained; their digests are in the attestation document.
