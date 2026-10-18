"""Reason codes that explain a hold or freeze without exposing anyone else's data.

A bank that holds a customer's payment has to say why: to the analyst who reviews it and,
in many jurisdictions, to the regulator. But the evidence for a cross-bank mule chain lives
partly in *other* banks' payments, which the bank must never see.

So the enclave reports only *families* of evidence, ranked by how much each pushed the score
up: "money moving in a cycle", "fan-in / fan-out", "unusual amount", ... Each code is a sum of
per-feature contributions (XGBoost's attribution for the same trees Snap ML scores with) over
a whole family, so it names a kind of pattern and never a counterparty, bank, account,
count or amount from another member.
"""

LABELS = {
    "cycle": "money moving in a cycle",
    "fan": "fan-in / fan-out around these accounts",
    "degree": "many counterparties in 24 h",
    "scatter_gather": "scatter-gather pattern",
    "account_profile": "account's 24 h amount profile",
    "amount": "unusual amount",
    "channel": "payment format or currency",
    "timing": "time of day or weekday",
    "route": "routing (same bank, self transfer, high-volume account)",
    "upstream": "money came from an already flagged payment",
}
# Families that can only be seen by pooling banks' payments in the enclave.
CROSS_BANK = {"cycle", "fan", "degree", "scatter_gather", "account_profile"}

UPSTREAM = {"code": "upstream", "label": LABELS["upstream"], "share": 1.0}


def reason_codes(contrib, groups: list[str], top: int = 3, min_share: float = 0.05) -> list[dict]:
    """Top evidence families pushing this payment's score up, with their share of the push."""
    totals: dict[str, float] = {}
    for g, v in zip(groups, contrib):
        totals[g] = totals.get(g, 0.0) + float(v)
    pos = {g: v for g, v in totals.items() if v > 0}
    total = sum(pos.values())
    if total <= 0:
        return []
    ranked = sorted(pos.items(), key=lambda kv: -kv[1])
    return [{"code": g, "label": LABELS.get(g, g), "share": round(v / total, 3)}
            for g, v in ranked[:top] if v / total >= min_share]
