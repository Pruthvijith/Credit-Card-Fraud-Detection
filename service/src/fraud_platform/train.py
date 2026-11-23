"""Leakage-controlled train/validation/test pipeline and cost-based alert threshold."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_platform.contracts import FEATURE_NAMES, FEATURE_VERSION, vector
from fraud_platform.synthetic import generate


def expected_cost(y_true: np.ndarray, probabilities: np.ndarray, threshold: float,
                  false_negative_cost: float, false_positive_cost: float) -> float:
    predicted = probabilities >= threshold
    return float(np.sum((y_true == 1) & (~predicted)) * false_negative_cost +
                 np.sum((y_true == 0) & predicted) * false_positive_cost)


def train(output: Path, rows: int = 4000, seed: int = 42,
          false_negative_cost: float = 500, false_positive_cost: float = 20) -> dict:
    if min(false_negative_cost, false_positive_cost) <= 0:
        raise ValueError("Business costs must be positive")
    examples, labels = generate(rows, seed)
    x = np.asarray([vector(e) for e in examples], dtype=float)
    y = np.asarray(labels)
    if np.bincount(y, minlength=2).min() < 3:
        raise ValueError("Insufficient positive/negative examples for stratified holdout")
    x_train, x_other, y_train, y_other = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(
        x_other, y_other, test_size=0.5, stratify=y_other, random_state=seed)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(x_train, y_train)
    val_probabilities = model.predict_proba(x_val)[:, 1]
    review_capacity = 0.10
    thresholds = [float(t) for t in np.append(np.unique(val_probabilities), 1.0)
                  if np.mean(val_probabilities >= t) <= review_capacity]
    # Threshold picked on validation only; never optimize against the test set.
    threshold = float(min(thresholds, key=lambda t: expected_cost(
        y_val, val_probabilities, float(t), false_negative_cost, false_positive_cost)))
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_prediction = test_probabilities >= threshold
    report = {
        "model_version": "fraud-logreg-v1",
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "data_source": "seeded synthetic generator, not real transactions",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_counts": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "positive_rate_test": float(np.mean(y_test)),
        "decision_threshold": threshold,
        "validation_review_capacity": review_capacity,
        "validation_alert_rate": float(np.mean(val_probabilities >= threshold)),
        "test_always_allow_cost_eur": float(y_test.sum() * false_negative_cost),
        "test_average_precision_baseline": float(np.mean(y_test)),
        "standardized_log_odds": dict(zip(FEATURE_NAMES, model[-1].coef_[0].tolist())),
        "test_false_negatives": int(np.sum((y_test == 1) & ~test_prediction)),
        "test_false_positives": int(np.sum((y_test == 0) & test_prediction)),
        "false_negative_cost_eur": false_negative_cost,
        "false_positive_cost_eur": false_positive_cost,
        "test_average_precision": float(average_precision_score(y_test, test_probabilities)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_precision": float(precision_score(y_test, test_prediction, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "test_alert_rate": float(np.mean(test_prediction)),
        "test_expected_cost_eur": expected_cost(y_test, test_probabilities, threshold,
                                                false_negative_cost, false_positive_cost),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    report["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output, rows=args.rows, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
