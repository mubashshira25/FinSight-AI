"""Run the complete FinSight AI analytics workflow."""
from pathlib import Path
import json
import pandas as pd
from generate_data import generate_transactions
from eda import run_eda
from modeling import run_model, run_anomaly_detection
from segmentation import run_segmentation

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
REPORTS = ROOT / "outputs" / "reports"


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("Generating synthetic transaction data...")
        generate_transactions().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} transactions")
    results = {
        "eda": run_eda(df),
        "fraud_model": run_model(df),
        "anomaly_detection": run_anomaly_detection(df),
        "customer_segmentation": run_segmentation(df),
    }
    (REPORTS / "final_project_summary.json").write_text(json.dumps(results, indent=2))
    print("\nFINISHED. Open outputs/figures and outputs/reports for results.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
