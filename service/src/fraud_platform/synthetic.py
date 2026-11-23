"""Seeded, entirely synthetic transactions. Not representative of real fraud rates."""

import math
import random

from fraud_platform.contracts import Transaction


def generate(count: int, seed: int = 42) -> tuple[list[Transaction], list[int]]:
    if count < 60:
        raise ValueError("At least 60 rows required for stratified splits")
    rng = random.Random(seed)
    rows: list[Transaction] = []
    labels: list[int] = []
    for i in range(count):
        amount = round(min(rng.lognormvariate(3.5, 1.0), 50000), 2)
        merchant_risk = round(rng.betavariate(2, 6), 5)
        hour = rng.randrange(24)
        age = rng.randint(1, 3500)
        velocity = min(int(rng.expovariate(1 / 3)), 10000)
        matches = rng.random() > 0.12
        row = Transaction(
            transaction_id=f"sim-{seed}-{i}",
            amount_eur=max(amount, 0.01),
            merchant_risk=merchant_risk,
            hour_utc=hour,
            card_age_days=age,
            transactions_24h=velocity,
            ip_country_matches_billing=matches,
        )
        # Plausible correlates for testing only. Not a model of actual fraud behaviour.
        risk = (
            -4.4
            + 2.5 * merchant_risk
            + 0.7 * (hour < 5)
            + 0.7 * (not matches)
            + 0.7 * (velocity >= 6)
            + 0.4 * (amount > 250)
            - 0.3 * math.log1p(age) / 8
        )
        probability = 1 / (1 + math.exp(-risk))
        rows.append(row)
        labels.append(int(rng.random() < probability))
    return rows, labels
