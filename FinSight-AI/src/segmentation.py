"""Customer behavior segmentation using K-Means."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
FIG = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "reports"


def run_segmentation(df: pd.DataFrame, clusters: int = 4) -> dict:
    FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    customer = df.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        average_transaction=("amount", "mean"),
        fraud_transaction_rate=("is_fraud", "mean"),
        payment_success_rate=("payment_success", "mean"),
        active_days=("timestamp", lambda x: pd.to_datetime(x).dt.date.nunique()),
    )
    features = ["transaction_count", "total_spend", "average_transaction", "fraud_transaction_rate", "payment_success_rate", "active_days"]
    # Log transform skewed monetary/frequency variables before scaling.
    model_data = customer[features].copy()
    for col in ["transaction_count", "total_spend", "average_transaction", "active_days"]:
        model_data[col] = (model_data[col].clip(lower=0) + 1).apply(__import__("numpy").log1p)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_data)
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=15)
    customer["segment"] = kmeans.fit_predict(scaled)
    score = silhouette_score(scaled, customer["segment"], sample_size=min(5_000, len(customer)), random_state=42)
    profile = customer.groupby("segment").agg(
        customers=("transaction_count", "size"),
        avg_transactions=("transaction_count", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_transaction_value=("average_transaction", "mean"),
        avg_fraud_rate=("fraud_transaction_rate", "mean"),
        avg_payment_success=("payment_success_rate", "mean"),
    ).round(3)
    customer.to_csv(REPORT / "customer_segments.csv")
    profile.to_csv(REPORT / "segment_profiles.csv")
    (REPORT / "segmentation_metrics.json").write_text(json.dumps({"clusters": clusters, "silhouette_score": round(float(score), 4)}, indent=2))

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(customer["transaction_count"], customer["total_spend"], c=customer["segment"], alpha=.55, s=18)
    ax.set(title="Customer segments", xlabel="Transaction count", ylabel="Total spend")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.2); fig.colorbar(scatter, ax=ax, label="Segment")
    fig.tight_layout(); fig.savefig(FIG / "customer_segments.png", dpi=150); plt.close(fig)
    return {"customers_segmented": int(len(customer)), "clusters": clusters, "silhouette_score": round(float(score), 4)}


if __name__ == "__main__":
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    print(run_segmentation(data))
