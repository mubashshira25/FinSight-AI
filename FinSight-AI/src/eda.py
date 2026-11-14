"""Exploratory data analysis and plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"


def run_eda(df: pd.DataFrame) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    summary = {
        "transactions": int(len(df)),
        "unique_customers": int(df["customer_id"].nunique()),
        "fraud_rate_pct": round(float(df["is_fraud"].mean() * 100), 3),
        "payment_success_rate_pct": round(float(df["payment_success"].mean() * 100), 2),
        "total_transaction_value": round(float(df["amount"].sum()), 2),
        "average_transaction_value": round(float(df["amount"].mean()), 2),
        "missing_cells": int(df.isna().sum().sum()),
    }

    daily = df.groupby("date").agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(daily.index), daily["transactions"], linewidth=1.4)
    ax.set(title="Daily transaction volume", xlabel="Date", ylabel="Transactions")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "daily_transaction_volume.png", dpi=150); plt.close(fig)

    fraud_by_method = df.groupby("payment_method")["is_fraud"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    fraud_by_method.plot(kind="bar", ax=ax)
    ax.set(title="Fraud rate by payment method", xlabel="Payment method", ylabel="Fraud rate (%)")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "fraud_by_payment_method.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(column="amount", by="is_fraud", ax=ax, showfliers=False)
    ax.set(title="Transaction amount distribution by fraud label", xlabel="Fraud label (0 = no, 1 = yes)", ylabel="Amount")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(FIGURES / "amount_by_fraud.png", dpi=150); plt.close(fig)

    country = df.groupby("country").agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"))
    country["fraud_rate_pct"] = country["fraud_rate"] * 100
    country.sort_values("fraud_rate_pct", ascending=False).to_csv(REPORTS / "country_risk_summary.csv")
    pd.DataFrame([summary]).to_csv(REPORTS / "kpi_summary.csv", index=False)
    df.groupby("failure_reason").size().sort_values(ascending=False).rename("count").to_csv(REPORTS / "failure_reason_summary.csv")
    return summary


if __name__ == "__main__":
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    print(run_eda(data))
