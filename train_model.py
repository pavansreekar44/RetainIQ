"""
train_model.py — XGBoost Churn Classifier Training Pipeline
============================================================

Trains a binary XGBoost classifier on the cleaned IBM Telco dataset and
persists **all artefacts** needed for inference on user-uploaded CSV files:

Saved artefacts (→ ``models/`` directory):
    • ``xgb_churn_model.json``   — trained XGBoost model (portable JSON format)
    • ``label_encoders.joblib``  — fitted sklearn LabelEncoders for categoricals
    • ``model_metadata.json``    — feature names, categories, thresholds, metrics

Usage:
    $ python train_model.py                       # train with defaults
    $ python train_model.py --data path/to.csv    # train on a different CSV
    $ python train_model.py --threshold 0.45      # custom decision threshold

The Streamlit app or any inference script only needs to load these three
files to predict on *any* industry CSV that has the same 7 feature columns.

Dataset contract (7 universal subscription features):
    Contract, tenure, MonthlyCharges, TotalCharges,
    PaymentMethod, TechSupport, PaperlessBilling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 7 universal subscription features (strict contract with mock_data.py)
FEATURE_COLUMNS: list[str] = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "TechSupport",
    "PaperlessBilling",
]

CATEGORICAL_COLUMNS: list[str] = [
    "Contract",
    "PaymentMethod",
    "TechSupport",
    "PaperlessBilling",
]

NUMERIC_COLUMNS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

TARGET_COLUMN: str = "Churn"

DEFAULT_DATA_PATH: str = os.path.join("dataset", "cleaned_customers.csv")
MODEL_DIR: str = "models"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from *df*."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual = set(df.columns)
    missing = required - actual
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(required)}. Got: {sorted(actual)}"
        )


def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data to encode (modified in-place for efficiency).
    encoders : dict or None
        Pre-fitted encoders to reuse (inference mode). If *None* and
        *fit=True*, new encoders are created and fitted.
    fit : bool
        If True, fit encoders on *df*. If False, only transform.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        The encoded DataFrame and the encoder dictionary.
    """
    if encoders is None:
        encoders = {}

    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen categories gracefully — map to -1
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x, _le=le, _k=known: (
                    _le.transform([x])[0] if x in _k else -1
                )
            )

    return df, encoders


def encode_target(series: pd.Series) -> pd.Series:
    """Convert Yes/No target to 1/0."""
    return series.map({"Yes": 1, "No": 0}).astype(int)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """End-to-end training pipeline.

    1. Load & validate CSV
    2. Encode categoricals + target
    3. Train/test split (stratified)
    4. Fit XGBoost with class-weight balancing
    5. Evaluate & print metrics
    6. Persist model, encoders, metadata

    Parameters
    ----------
    data_path : str
        Path to the cleaned CSV.
    test_size : float
        Fraction of data reserved for evaluation.
    random_state : int
        Seed for reproducibility.
    threshold : float
        Probability threshold for the "high risk" label.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics dictionary.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"\n📂  Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    validate_columns(df)

    # ── 2. Encode ────────────────────────────────────────────────────────
    df, encoders = encode_categoricals(df, fit=True)
    y = encode_target(df[TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]

    churn_rate = y.mean()
    print(f"    Churn rate: {churn_rate:.1%}  ({y.sum():,} / {len(y):,})")

    # ── 3. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    # scale_pos_weight compensates for the class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        use_label_encoder=False,
    )

    print("\n🏋️  Training XGBoost …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "threshold": threshold,
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate": round(churn_rate, 4),
        "scale_pos_weight": round(scale_pos_weight, 4),
    }

    print("\n📊  Evaluation Results")
    print("─" * 50)
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Threshold: {metrics['threshold']}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # ── 6. Save artefacts ────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "xgb_churn_model.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoders.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")

    # 6a. Model — joblib handles the full sklearn-wrapped XGBClassifier
    joblib.dump(model, model_path)
    print(f"💾  Model saved:    {model_path}")

    # 6b. Label encoders
    joblib.dump(encoders, encoder_path)
    print(f"💾  Encoders saved: {encoder_path}")

    # 6c. Metadata — everything the inference code needs to reconstruct
    #     the pipeline without reading this training script.
    encoder_classes: dict[str, list[str]] = {
        col: le.classes_.tolist() for col, le in encoders.items()
    }

    metadata: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": TARGET_COLUMN,
        "encoder_classes": encoder_classes,
        "metrics": metrics,
        "model_file": "xgb_churn_model.joblib",
        "encoder_file": "label_encoders.joblib",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾  Metadata saved: {metadata_path}")

    # ── Feature importance ───────────────────────────────────────────────
    importances = model.feature_importances_
    importance_pairs = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n🔍  Feature Importance (gain)")
    print("─" * 50)
    for feat, imp in importance_pairs:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<20s} {imp:.4f}  {bar}")

    print(f"\n✅  All artefacts saved to ./{MODEL_DIR}/")
    print("    The Streamlit app can now load these for inference.\n")

    return metrics


# ---------------------------------------------------------------------------
# Inference helper (used by the Streamlit app)
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str = MODEL_DIR,
) -> tuple[XGBClassifier, dict[str, LabelEncoder], dict[str, Any]]:
    """Load the trained model, encoders, and metadata from disk.

    Returns
    -------
    tuple[XGBClassifier, dict[str, LabelEncoder], dict[str, Any]]
        (model, encoders, metadata)
    """
    model_path = os.path.join(model_dir, "xgb_churn_model.joblib")
    encoder_path = os.path.join(model_dir, "label_encoders.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    for path in [model_path, encoder_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artefact: {path}. Run `python train_model.py` first."
            )

    model: XGBClassifier = joblib.load(model_path)

    encoders: dict[str, LabelEncoder] = joblib.load(encoder_path)

    with open(metadata_path) as f:
        metadata: dict[str, Any] = json.load(f)

    return model, encoders, metadata


def predict_from_dataframe(
    df: pd.DataFrame,
    model: XGBClassifier | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run churn predictions on an arbitrary user-uploaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 7 feature columns. May optionally contain
        ``customerID`` and/or ``Churn`` (both are preserved if present).
    model, encoders, metadata : optional
        Pre-loaded artefacts. If *None*, loaded from disk.
    threshold : float or None
        Override the training threshold stored in metadata.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with:
        * ``churn_risk_score`` — float [0, 1]
        * ``is_high_risk``     — bool
    """
    if model is None or encoders is None or metadata is None:
        model, encoders, metadata = load_model()

    if threshold is None:
        threshold = metadata["metrics"]["threshold"]

    # Validate feature columns exist
    feature_cols: list[str] = metadata["feature_columns"]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(feature_cols)}"
        )

    # Encode categoricals using the saved encoders (fit=False)
    df_encoded, _ = encode_categoricals(df, encoders=encoders, fit=False)

    X = df_encoded[feature_cols]
    probas = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["churn_risk_score"] = np.round(probas, 4)
    result["is_high_risk"] = probas >= threshold

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the XGBoost churn classifier and save model artefacts."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the cleaned CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for high-risk label (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.seed,
        threshold=args.threshold,
    )
