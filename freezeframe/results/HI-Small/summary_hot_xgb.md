# HI-Small — alone vs together (hot features, xgb)

Together, all test transactions: F1 **0.6515** (precision 0.852, recall 0.528), recall@0.5% 0.679, laundered $ stopped 99.7%

## Per-pattern recall (together, test split)

| Pattern | Positives | Recall |
|---|---|---|
| BIPARTITE | 74 | 0.581 |
| CYCLE | 109 | 0.596 |
| FAN-IN | 137 | 0.752 |
| FAN-OUT | 140 | 0.800 |
| GATHER-SCATTER | 401 | 0.743 |
| RANDOM | 86 | 0.733 |
| SCATTER-GATHER | 252 | 0.726 |
| STACK | 140 | 0.564 |
| (unpatterned) | 458 | 0.004 |
