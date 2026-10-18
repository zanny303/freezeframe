# HI-Small — alone vs together (full features, xgb)

Together, all test transactions: F1 **0.6304** (precision 0.915, recall 0.481), recall@0.5% 0.668, laundered $ stopped 99.7%

## Seed 0

Overall alone F1 **0.5612** vs together **0.6304**; laundered $ stopped alone 99.7% vs together 99.7%

| Member | Banks | Test positives | F1 alone | F1 together | Recall@0.5% alone | Recall@0.5% together | $ stopped alone | $ stopped together |
|---|---|---|---|---|---|---|---|---|
| 0 | 6160 | 359 | 0.3605 | 0.4195 | 0.401 | 0.418 | 99.1% | 99.4% |
| 1 | 6023 | 448 | 0.5930 | 0.6657 | 0.658 | 0.743 | 99.8% | 99.8% |
| 2 | 6218 | 313 | 0.6046 | 0.6653 | 0.658 | 0.719 | 99.8% | 99.7% |
| 3 | 5998 | 289 | 0.6208 | 0.6466 | 0.647 | 0.716 | 99.8% | 99.7% |
| 4 | 6129 | 388 | 0.6094 | 0.7118 | 0.665 | 0.732 | 99.9% | 99.8% |

## Seed 1

Overall alone F1 **0.5349** vs together **0.6304**; laundered $ stopped alone 99.7% vs together 99.7%

| Member | Banks | Test positives | F1 alone | F1 together | Recall@0.5% alone | Recall@0.5% together | $ stopped alone | $ stopped together |
|---|---|---|---|---|---|---|---|---|
| 0 | 6203 | 417 | 0.4745 | 0.4834 | 0.489 | 0.477 | 99.3% | 99.4% |
| 1 | 6059 | 419 | 0.5126 | 0.6954 | 0.697 | 0.759 | 99.7% | 99.7% |
| 2 | 6134 | 392 | 0.5331 | 0.6667 | 0.666 | 0.717 | 99.8% | 99.8% |
| 3 | 6058 | 230 | 0.5536 | 0.6740 | 0.643 | 0.709 | 99.9% | 99.9% |
| 4 | 6074 | 339 | 0.6215 | 0.6344 | 0.655 | 0.702 | 99.8% | 99.7% |

## Seed 2

Overall alone F1 **0.5799** vs together **0.6304**; laundered $ stopped alone 99.7% vs together 99.7%

| Member | Banks | Test positives | F1 alone | F1 together | Recall@0.5% alone | Recall@0.5% together | $ stopped alone | $ stopped together |
|---|---|---|---|---|---|---|---|---|
| 0 | 6131 | 319 | 0.5981 | 0.6453 | 0.646 | 0.687 | 99.8% | 99.8% |
| 1 | 6173 | 296 | 0.5522 | 0.6521 | 0.662 | 0.720 | 99.7% | 99.5% |
| 2 | 6069 | 367 | 0.6265 | 0.6539 | 0.700 | 0.747 | 99.7% | 99.6% |
| 3 | 5968 | 383 | 0.6299 | 0.6811 | 0.684 | 0.708 | 99.8% | 99.7% |
| 4 | 6187 | 432 | 0.4939 | 0.5303 | 0.470 | 0.521 | 99.6% | 99.7% |

Alone overall F1 across seeds: mean 0.5587, min 0.5349, max 0.5799

## Per-pattern recall (together, test split)

| Pattern | Positives | Recall |
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
