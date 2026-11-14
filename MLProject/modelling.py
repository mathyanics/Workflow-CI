"""Train baseline credit-default classifier and log with MLflow autolog."""

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("credit_default")


def build_pipeline() -> Pipeline:
    """Construct data preprocessing and baseline classification pipeline."""
    numeric_features = ["balance", "income"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_features = ["student"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def main(data_path: str = "credit_default_preprocessed.csv") -> dict[str, float]:
    """Train baseline credit default model and log evaluation metrics to MLflow."""
    df = pd.read_csv(data_path)
    y = df.pop("default").astype(int)
    X = df[["student", "balance", "income"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_pipeline()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    project_run_id = os.getenv("MLFLOW_RUN_ID")
    if not project_run_id:
        mlflow.set_experiment("Credit Default Risk")

    mlflow.sklearn.autolog(log_models=True)

    run_context = nullcontext() if project_run_id else mlflow.start_run()
    with run_context:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
        }

        mlflow.log_metrics(metrics)
        LOG.info("Holdout metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline credit default risk model."
    )
    parser.add_argument(
        "--data",
        default="credit_default_preprocessed.csv",
        help="Path to preprocessed CSV dataset.",
    )
    args = parser.parse_args()
    main(args.data)
