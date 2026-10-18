# FreezeFrame

**Catching scam money across banks without sharing customer data.** This is an IBM Z Datathon 2026 entry, in the track "AI Secured: Innovation Without Exposure".

Scam money hops through chains of money mules across several banks within minutes. Each bank sees only its own slice of the chain. In FreezeFrame, each bank streams its own payments, encrypted, into a shared enclave on IBM LinuxONE:

1. Account numbers become keyed pseudonyms before any processing.
2. IBM's Graph Feature Preprocessor mines fan-in, fan-out, cycle, and scatter-gather patterns as payments arrive.
3. A gradient-boosted model, scored through Snap ML (on the IBM Z AI accelerator when one is present), gives each payment a calibrated probability.
4. A cost rule decides: release, hold for review, or ask the receiving bank to freeze.

Each decision goes back only to the bank that owns the payment, signed by the enclave, with privacy-safe reason codes ("money moving in a cycle", "fan-in / fan-out") that name a kind of pattern, never another bank's customer.

**Why this is "AI Secured: Innovation Without Exposure":** the AI only works *because* it sees all banks' payments together (pooled F1 0.630 vs 0.559 alone on HI-Small), yet no bank, operator or attacker on the wire sees another bank's data. Every message is signed and sealed with replay protection, banks verify the enclave's attestation before sending, and the dashboard lets you attack the boundary live. See [Security model](#security-model).

The pitch deck and technical brief are in `submission/`; what comes next is in the [Roadmap](#roadmap).

## Results

<!-- RESULTS -->

### HI-Small: caught alone vs together (full features, xgb)

Test split: 1,015,669 payments, 1,797 laundering. Five consortium members, banks dealt at random; each seed is a different deal.

| | Minority-class F1 | Precision | Recall | Laundered $ stopped |
|---|---|---|---|---|
| **Together (enclave)** | **0.6304** | 0.915 | 0.481 | 99.7% |
| Alone, seed 0 | 0.5612 | 0.740 | 0.452 | 99.7% |
| Alone, seed 1 | 0.5349 | 0.814 | 0.398 | 99.7% |
| Alone, seed 2 | 0.5799 | 0.804 | 0.454 | 99.7% |

Pooling raises F1 from **0.5587** (mean alone, range 0.5349–0.5799) to **0.6304**. The published GFP + XGBoost baseline on HI-Small is 0.6323 (Blanuša et al.).

Per member, seed 0 (same owned test payments in both columns):

| Member | Banks | Test laundering | F1 alone | F1 together | Recall@0.5% alone | Recall@0.5% together |
|---|---|---|---|---|---|---|
| A | 6,160 | 359 | 0.3605 | 0.4195 | 0.401 | 0.418 |
| B | 6,023 | 448 | 0.5930 | 0.6657 | 0.658 | 0.743 |
| C | 6,218 | 313 | 0.6046 | 0.6653 | 0.658 | 0.719 |
| D | 5,998 | 289 | 0.6208 | 0.6466 | 0.647 | 0.716 |
| E | 6,129 | 388 | 0.6094 | 0.7118 | 0.665 | 0.732 |

What the decision rule does on the test split (counts of payments):

| | Laundering frozen | Legit frozen | Laundering stopped (hold or freeze) | Legit held |
|---|---|---|---|---|
| **Together** | 611 of 1797 | 12 | 1247 | 17,962 |
| Alone, seed 0 | 389 of 1797 | 11 | 1217 | 18,225 |
| Alone, seed 1 | 353 of 1797 | 4 | 1218 | 18,602 |
| Alone, seed 2 | 344 of 1797 | 6 | 1232 | 17,259 |

Laundered *dollars* stopped is about 99.7% in every arm: the hold rule (p × amount > $100) catches nearly all large laundering payments either way, so the dollar metric does not separate the arms. It also holds about 1.8% of all payments, more than a 0.5% review budget; tuning the review cost is on the Roadmap.

Recall by laundering pattern (together, test split):

| Pattern | Laundering payments | Recall |
|---|---|---|
| BIPARTITE | 74 | 0.541 |
| CYCLE | 109 | 0.505 |
| FAN-IN | 137 | 0.693 |
| FAN-OUT | 140 | 0.743 |
| GATHER-SCATTER | 401 | 0.698 |
| RANDOM | 86 | 0.616 |
| SCATTER-GATHER | 252 | 0.667 |
| STACK | 140 | 0.486 |
| (unpatterned) | 458 | 0.002 |

Live-path (hot tier, no scatter-gather) model, together: F1 0.6515, laundered $ stopped 99.7%. It scores *higher* than the full tier, so on HI-Small with these windows scatter-gather adds no signal; the alone-vs-together comparison above was run on the full tier only.

### Streaming latency: warm tier process (inprocess bus, x86_64, macOS)

Scoring: `compress_trees` · enclave attestation: `simulated` · budget 50 ms p99 from scheduled arrival to published decision (includes a 5 ms bank-side batching linger).

| Speed vs real time | Payments | Achieved / s | p50 ms | p95 ms | p99 ms | Mean micro-batch |
|---|---|---|---|---|---|---|
| 1× | 301 | 5 | 12.4 | 19.4 | 27.2 | 1.0 |
| 100× | 24,344 | 539 | 11.0 | 16.2 | 20.9 | 3.5 |
| 300× | 111,224 | 2,447 | 16.0 | 29.8 | 135.2 | 20.7 |
| 1000× | 333,218 | 7,351 | 39.7 | 396.9 | 616.2 | 160.4 |

Highest rate with p99 within budget: **539 payments/s**.

### Streaming latency: warm tier off (inprocess bus, x86_64, macOS)

Scoring: `compress_trees` · enclave attestation: `simulated` · budget 50 ms p99 from scheduled arrival to published decision (includes a 5 ms bank-side batching linger).

| Speed vs real time | Payments | Achieved / s | p50 ms | p95 ms | p99 ms | Mean micro-batch |
|---|---|---|---|---|---|---|
| 1× | 301 | 5 | 12.5 | 19.2 | 23.7 | 1.0 |
| 100× | 24,344 | 539 | 10.8 | 15.8 | 19.9 | 3.5 |
| 300× | 111,224 | 2,448 | 14.1 | 22.8 | 47.8 | 18.3 |
| 1000× | 333,218 | 7,353 | 25.3 | 242.8 | 515.1 | 99.1 |

Highest rate with p99 within budget: **2,448 payments/s**.

### Demo replay (demo/app.py)

A real 8-payment CYCLE laundering attempt from the test split, crossing all five members, replayed through sealed bank clients with live background traffic around each hop. All 8 payments were held by the live hot path; one (hop 5) only because its money came from a flagged payment (watchlist, `upstream_flag`). In the experiment's predictions the pooled model stops all 8 hops; the members alone miss 3 of them.

_Measured on the dev laptop (Apple M2, x86_64 Python under Rosetta) unless the heading says otherwise. Not yet measured on LinuxONE._

<!-- /RESULTS -->

## Quick start

**Two-minute demo on synthetic data (no Kaggle download, no 1-hour experiment):**

```bash
pip install -e '.[dev,xgb]'
./scripts/quick_demo.sh          # synthetic data -> experiment (~4 min on 2 cores) -> dashboard on :8000
```

The synthetic run is a smoke test: the page labels it "synthetic, not results". The real numbers above come from HI-Small.

**Full run on IBM's AMLworld data:**

```bash
# Apple Silicon: Snap ML needs an x86_64 Python under Rosetta (the script handles libomp)
./scripts/setup_mac.sh
PY="arch -x86_64 .venv/bin/python"          # on Linux: pip install -e '.[dev,xgb]'; PY=python

# data (no Kaggle login needed)
mkdir -p data && for f in HI-Small_Trans.csv HI-Small_Patterns.txt; do
  curl -L -o data/$f https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/$f
done

$PY -m pytest                                # 31 tests (the Kafka test needs FF_KAFKA=host:port)
./scripts/run_experiments.sh HI-Small        # alone vs together + live-path model (~1 h)
$PY -m bench.latency --speeds 1 100 1000     # streaming latency
$PY -m demo.app                              # live dashboard on http://localhost:8000 (PORT=… to change)
```

## Live demo dashboard

`demo/app.py` replays a real laundering attempt through the running system. It's an 8-payment cycle from the test split that crosses all five banks. The payments go through sealed bank clients, the enclave's hot path, and the warm tier. Startup takes about 2–3 minutes while the enclave loads the graph; then open the page and press **Play**. The replay takes about a minute.

The page reads top to bottom:

1. **Headline results:** F1 for each bank alone vs together, and laundering payments frozen automatically, both from `results/`.
2. **The money's route:** the accounts in the order the money moved, coloured by bank, with the last payment returning to the first account. Below it is a timeline whose cursor follows the payment being replayed.
3. **Alone vs together, one row per payment.** Each row reads across:
   - on the left, what the paying bank decides with only its own view (from the experiment's models)
   - in the middle, the payment itself
   - on the right, the live enclave decision as it arrives, with its reason, and any freeze or hold notice sent to the receiving bank

   A payment held only because its money came from an already-flagged payment is labelled "held at the next hop".
4. **F1 by bank** and the **streaming benchmark** table.
5. **Try to break the boundary.** Five attack buttons run against the live system: replay a captured bank message, submit as another bank without its key, have a real member submit another bank's payment, flip bytes in a sealed message, and forge a "release" decision on a bank's topic. Each is blocked (by the enclave, or by the bank for the forged decision) and logged on the page. None reaches the graph, so you can use them before, during or after the replay.
6. **Live console** pinned to the bottom: payments per second, p50 and p99 decision latency, payments scored, holds, freeze requests, next-hop holds, warm-tier escalations, and attacks blocked.

Reading the dashboard:

- **Each hop is replayed live,** at 40× real time, starting a few minutes before the payment. The hours between hops are *fast-forwarded*: those payments are loaded into the graph, so its windows stay correct, but they aren't scored or shown. Each skip appears as a divider between the two payments it separates, with its length and payment count.
- **Labels:** "enclave: simulated" and "scoring: CPU" stay on screen unless the server runs under Secure Execution or on the IBM Z AI accelerator.
- **Each bank only sees its own decisions.** The page is the demo operator's merged view of those per-bank topics.
- **One replay per server start.** A second one would feed the same payments into the graph again, so restart the server to replay. Refreshing the page catches up on a replay in progress.

Keys: **Space** plays; **← →** steps through decided payments and highlights them on the route; **Esc** returns to live.

`prototypes/` holds the six design directions explored for this page. They're standalone, play a recorded run, and are browsable from `prototypes/index.html`. The dashboard is prototype 06, "Story × Split".

## How it works

```
bank client ──seal(payment, enclave key)──▶ payments.in  (one partition, time-ordered)
                                                │
                                   enclave/stream.py  HotConsumer (micro-batches)
       ownership check → pseudonymise → GFP update (per timestamp) → booster → calibrate → decide
                                                │                         ▲
                     decisions.<bank>  sealed to that bank's key          │ watchlist
                     notices.<bank>    freeze requests for its own accts  │
                                                                 enclave/warm.py (every few s:
                                                                 full patterns, escalations)
```

| Piece | File |
|---|---|
| CSV to typed parquet, USD rates from the data, time split, pattern labels | `freezeframe/ingest.py` |
| HMAC pseudonyms (key stays in the enclave) | `freezeframe/pseudonym.py` |
| Random bank-to-member assignment, views, owners | `freezeframe/consortium.py` |
| Raw and GFP features, hub splitting, timestamp batching | `freezeframe/features.py` |
| XGBoost trained, then imported into Snap ML for scoring (or Snap ML's own booster) | `freezeframe/model.py` |
| Isotonic calibration; release / hold / freeze rule | `freezeframe/decide.py` |
| Privacy-safe reason codes for held / frozen payments | `freezeframe/explain.py` |
| Sign-then-seal envelopes, replay guard | `freezeframe/enclave/crypto.py` |
| Member registry, one-time enrollment tokens, key rotation | `freezeframe/enclave/identity.py` |
| Attestation document; bank-side verification | `freezeframe/enclave/attest.py` |
| Synthetic AMLworld-shaped data for the quick demo | `freezeframe/synth.py` |
| Alone-vs-together experiment | `freezeframe/experiment.py` |
| Enclave engine, warm tier, transports, HTTP API, crypto | `freezeframe/enclave/` |
| Streaming latency benchmark | `bench/latency.py` |
| Live demo dashboard (server, page) | `demo/app.py`, `demo/static/index.html` |
| Dashboard design prototypes | `prototypes/` |
| LinuxONE steps, Dockerfile | `deploy/` |

### Decisions

- **Hold** when the expected laundered dollars (probability × amount) are more than the cost of a review ($100).
- **Freeze** when the calibrated probability clears the threshold that gives 95% precision on the tune split. The receiving bank then gets a notice naming only *its own* account.
- **Hold at hop 2.** When a payment is held or frozen, the receiving account is watched for 72 h. Payments *out* of that account are held, even if their own score is low. Hub accounts (more than 10k train edges, e.g. payment processors) are never watched, so one held payment into a hub can't hold the hub's traffic for three days. Expired entries are pruned.
- **Reason codes.** Held and frozen payments carry the top evidence families behind the score (cycle, fan-in/out, degree, scatter-gather, account profile, amount, channel, timing), each with its share. They are sums of XGBoost attributions over whole families of features, computed on the same trees Snap ML scores with, so they say *what kind* of pattern, never who or which bank.

### Things we learned about the Graph Feature Preprocessor

- `fit_transform` inserts each edge twice: once in `fit`, again in `transform`. This doubles every count, so we only call `transform`.
- Within one `transform` call, every edge sees the whole batch, including later edges. We cut batches at timestamp boundaries, so the batch features and the live features are identical, with no look-ahead. `tests/test_features.py` checks this.
- Two HI-Small accounts have more than 100k edges each, and they make pattern mining blow up: a full 5M-row run did not finish in 30 minutes. *Hub splitting* gives each edge of an account with more than 10k train edges its own vertex, and adds is-hub flags. With it, all patterns over 5M rows take about 90 s on a laptop.
- The `*_tw` windows are not hard look-back bounds, so the enclave primes its graph from full history (see the Roadmap).

## Security model

Threat model: banks are honest-but-curious about each other, the network and bus are hostile, and the enclave host operator should learn nothing (IBM Secure Execution's job). Details are in `SECURITY.md`.

| Attack | Defence | Where | Test |
|---|---|---|---|
| Read another bank's payments or decisions | Sealed boxes to the recipient's key; decisions only to the owner | `crypto.py`, `engine.py` | `test_enclave.py` |
| Submit payments as another bank | Ed25519 signature checked against the key registered for that member | `crypto.py`, `stream.py`, `service.py` | `test_envelope_checks` |
| Submit another bank's payment while authenticated | Ownership check: sending bank must belong to the submitting member | `engine.py` | `test_cannot_submit_another_members_payment` |
| Replay a captured message | Nonce + 120 s freshness window, bounded cache | `crypto.ReplayGuard` | `test_envelope_checks`, `test_replay_guard_bounded` |
| Forge a "release" decision on a bank's topic | Banks accept only enclave-signed messages | `BankClient.open` | `test_consumer_survives_attacks_and_counts_them` |
| Overwrite a member's key over HTTP (was possible) | One-time enrollment token; rotation needs the current key | `identity.py`, `service.py` | `test_http_enrollment_and_notices_are_authenticated` |
| Drain another bank's freeze notices (was possible) | `POST /v1/notices` must be signed by the member | `service.py` | same |
| Crash the enclave with junk on `payments.in` (was possible) | Every message and payment validated; failures counted, never fatal | `HotConsumer`, `Engine._frame` | `test_consumer_survives_attacks_and_counts_them`, `test_malformed_payment_does_not_sink_the_batch` |
| Talk to a fake or modified enclave | Attestation doc: keys + code measurement + model digests, verified by each bank | `attest.py`, `BankClient.from_attestation` | `test_attestation_document` |
| Learn raw account numbers from features, logs, models | HMAC pseudonyms before processing; tests scan for raw strings | `pseudonym.py` | `test_no_raw_account_inside_enclave` |

Every refusal is counted under `security` in `/v1/metrics` and shown live on the dashboard.

## Roadmap

1. **Run on LinuxONE Community Cloud** (s390x): tests, the HI-Small experiment, the latency benchmark and the dashboard; commit `results/*/latency_*_s390x.json`. Steps are in `deploy/linuxone.md`.
2. **IBM Z Integrated Accelerator for AI:** benchmark Snap ML scoring with `FF_TREE_FORMAT=zdnn_tensors` vs `compress_trees`, at several batch sizes.
3. **Secure Execution:** run the enclave image under IBM Confidential Computing Container Runtime (Hyper Protect), put the keys' hash in the hardware attestation record, and switch banks to `allow_simulated=False`.
4. **IBM Event Streams:** per-member ACLs (each bank writes only `payments.in` and reads only its own topics), and a consumer thread instead of polling.
5. **Scale:** HI-Medium (32M payments) with memory-mapped features; report GFP throughput and peak memory.
6. **Fast restart:** a pseudonymous edge log replayed on restart, instead of re-priming from history.
7. **Graph windows:** establish how GFP's time windows evict edges, to prime from a bounded window.
8. **Model:** hyperparameter search, a separate early-stopping split, and a review cost set so holds match the review budget; benchmark Snap ML's own booster; run LI-Small.
9. **Rerun the latency benchmark** with signed envelopes and reason codes on.

## Notes on the data and measurements

- Every number in this README comes from files in `results/`, which also record the machine each run used.
- The data is IBM's synthetic AMLworld (Altman et al., 2023). No real customer data is used.
