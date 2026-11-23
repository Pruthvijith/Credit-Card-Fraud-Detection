"""Versioned feature contract. Never accept raw card identifiers or customer PII."""

from math import log1p

from pydantic import BaseModel, ConfigDict, Field

FEATURE_VERSION = "transaction-features-v1"
FEATURE_NAMES = (
    "log_amount",
    "merchant_risk",
    "night_transaction",
    "log_card_age_days",
    "log_txn_24h",
    "country_mismatch",
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=80)
    amount_eur: float = Field(gt=0, le=1_000_000, allow_inf_nan=False)
    merchant_risk: float = Field(ge=0, le=1, allow_inf_nan=False)
    hour_utc: int = Field(ge=0, le=23)
    card_age_days: int = Field(ge=0, le=36500)
    transactions_24h: int = Field(ge=0, le=10000)
    ip_country_matches_billing: bool


def vector(transaction: Transaction) -> list[float]:
    """Keep ordering in one place to avoid train/serve skew."""
    return [
        log1p(transaction.amount_eur),
        transaction.merchant_risk,
        float(transaction.hour_utc < 5),
        log1p(transaction.card_age_days),
        log1p(transaction.transactions_24h),
        float(not transaction.ip_country_matches_billing),
    ]
