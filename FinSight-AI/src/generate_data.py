"""Generate a reproducible synthetic payments dataset for FinSight AI.

All records are synthetic. The generator is for portfolio/learning use and does
not represent real PayPal data or a production fraud process.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def generate_transactions(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = np.datetime64("2025-01-01T00:00:00")
    minutes = rng.integers(0, 365 * 24 * 60, size=n)
    timestamps = start + minutes.astype("timedelta64[m]")

    customer_count = max(2_500, n // 8)
    customer_ids = rng.integers(1, customer_count + 1, size=n)
    customer_age_days = rng.integers(1, 2_500, size=customer_count + 1)[customer_ids]
    customer_tenure = rng.integers(1, 1_500, size=customer_count + 1)[customer_ids]

    countries = np.array(["US", "GB", "IN", "DE", "CA", "FR", "BR", "NG", "AU", "SG"])
    country_probs = np.array([.30, .12, .18, .08, .08, .06, .06, .04, .05, .03])
    country = rng.choice(countries, size=n, p=country_probs)
    payment_methods = np.array(["CARD", "WALLET", "BANK_TRANSFER", "UPI"])
    payment_method = rng.choice(payment_methods, size=n, p=[.46, .25, .12, .17])
    device = rng.choice(["MOBILE", "DESKTOP", "TABLET"], size=n, p=[.64, .31, .05])

    # Heavy-tailed amounts are typical of payment data; cap extreme synthetic values.
    amount = np.clip(rng.lognormal(mean=3.35, sigma=1.05, size=n), 1.0, 5_000).round(2)
    hour = pd.DatetimeIndex(timestamps).hour.to_numpy()
    day_of_week = pd.DatetimeIndex(timestamps).dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)

    # Build risk propensity from observable behavioral features, then sample labels.
    high_risk_country = np.isin(country, ["NG", "BR"]).astype(int)
    night = ((hour <= 4) | (hour >= 23)).astype(int)
    new_account = (customer_age_days < 30).astype(int)
    high_amount = (amount > 700).astype(int)
    device_switch_proxy = (device == "TABLET").astype(int)
    risk_logit = (
        -4.6 + 1.55 * high_risk_country + 1.35 * night + 1.40 * new_account
        + 1.45 * high_amount + .65 * device_switch_proxy
        + .45 * (payment_method == "CARD").astype(int)
        + .00018 * amount
    )
    probability = 1 / (1 + np.exp(-risk_logit))
    # Add a small random fraud cluster to create learnable but imperfect patterns.
    hidden_cluster = rng.random(n) < .002
    probability = np.clip(probability + hidden_cluster * .18, .002, .65)
    is_fraud = rng.binomial(1, probability)

    # Transaction outcome is separate from fraud label; failures are not necessarily fraud.
    failure_probability = .035 + .025 * (payment_method == "BANK_TRANSFER") + .018 * (amount > 1_000)
    payment_success = rng.binomial(1, 1 - np.clip(failure_probability, 0, .25))
    failure_reasons = np.array(["INSUFFICIENT_FUNDS", "DO_NOT_HONOR", "NETWORK_ERROR", "AUTHENTICATION_FAILED"])
    failure_reason = np.where(payment_success == 1, "NONE", rng.choice(failure_reasons, size=n))

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(1, n + 1)],
        "customer_id": [f"CUST{x:06d}" for x in customer_ids],
        "timestamp": pd.to_datetime(timestamps),
        "amount": amount,
        "currency": rng.choice(["USD", "EUR", "GBP", "INR"], size=n, p=[.48, .20, .12, .20]),
        "country": country,
        "payment_method": payment_method,
        "device": device,
        "customer_age_days": customer_age_days,
        "customer_tenure_days": customer_tenure,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "payment_success": payment_success,
        "failure_reason": failure_reason,
        "is_fraud": is_fraud,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_transactions()
    path = DATA_DIR / "transactions.csv"
    data.to_csv(path, index=False)
    print(f"Created {len(data):,} synthetic transactions: {path}")
    print(f"Fraud prevalence: {data['is_fraud'].mean():.2%}")
