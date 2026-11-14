"""Fraud classification with leakage-aware preprocessing and evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORT = OUT / "reports"

FEATURES_NUM = ["amount", "customer_age_days", "customer_tenure_days", "hour", "day_of_week", "is_weekend"]
FEATURES_CAT = ["currency", "country", "payment_method", "device"]
TARGET = "is_fraud"


def run_model(df: pd.DataFrame) -> dict:
    FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    # Timestamp and IDs are excluded to avoid memorizing identifiers or leaking chronology.
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET].astype(int)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=.20, random_state=43, stratify=y_train_full
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("num", numeric, FEATURES_NUM), ("cat", categorical, FEATURES_CAT)])
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                                   random_state=42, n_jobs=-1, max_features="sqrt")
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)
    # Select a decision threshold on validation data, not on the held-out test set.
    validation_probabilities = pipeline.predict_proba(X_valid)[:, 1]
    val_precision, val_recall, val_thresholds = precision_recall_curve(y_valid, validation_probabilities)
    val_f1 = (2 * val_precision[:-1] * val_recall[:-1]) / (val_precision[:-1] + val_recall[:-1] + 1e-12)
    threshold = float(val_thresholds[int(np.argmax(val_f1))]) if len(val_thresholds) else .5
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    metrics = {
        "test_rows": int(len(y_test)), "fraud_prevalence_pct": round(float(y_test.mean() * 100), 3),
        "threshold_selected_on_validation": round(threshold, 5), "precision_fraud": round(report["1"]["precision"], 4),
        "recall_fraud": round(report["1"]["recall"], 4), "f1_fraud": round(report["1"]["f1-score"], 4),
        "pr_auc": round(float(pr_auc), 4), "roc_auc": round(float(roc_auc), 4),
        "accuracy": round(report["accuracy"], 4),
    }
    (REPORT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORT / "classification_report.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["Legitimate", "Fraud"]).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Fraud model — confusion matrix")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "precision_recall_curve.png", dpi=150); plt.close(fig)

    # Permutation-style importance using the model's native feature importances after encoding.
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    importance = pd.Series(importances, index=names).sort_values(ascending=False).head(15)
    importance.sort_values().plot(kind="barh", figsize=(9, 6), title="Top model feature importances")
    plt.xlabel("Random Forest impurity importance"); plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()

    scored = X_test.copy()
    scored["actual_fraud"] = y_test.to_numpy()
    scored["fraud_probability"] = probabilities
    scored["predicted_fraud"] = predictions
    scored.to_csv(REPORT / "test_set_predictions.csv", index=False)
    return metrics


def run_anomaly_detection(df: pd.DataFrame) -> dict:
    # Unsupervised anomaly model; it is not trained on the fraud labels.
    sample = df[["amount", "customer_age_days", "customer_tenure_days", "hour"]].copy()
    sample = sample.replace([np.inf, -np.inf], np.nan).fillna(sample.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample)
    detector = IsolationForest(n_estimators=200, contamination=.02, random_state=42, n_jobs=-1)
    labels = detector.fit_predict(scaled)
    scores = -detector.decision_function(scaled)
    result = df[["transaction_id", "customer_id", "timestamp", "amount", "is_fraud"]].copy()
    result["anomaly_flag"] = (labels == -1).astype(int)
    result["anomaly_score"] = scores
    result.nlargest(500, "anomaly_score").to_csv(REPORT / "top_anomalies.csv", index=False)
    flagged = result[result["anomaly_flag"] == 1]
    metrics = {"flagged_transactions": int(len(flagged)), "flagged_rate_pct": round(float(len(flagged) / len(result) * 100), 2),
               "fraud_rate_among_flagged_pct": round(float(flagged["is_fraud"].mean() * 100), 2)}
    (REPORT / "anomaly_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    print("Supervised model:", run_model(data))
    print("Anomaly detection:", run_anomaly_detection(data))
