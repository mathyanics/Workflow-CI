"""Tune logistic regression and manually log metrics/artifacts to MLflow."""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("tuning")


def build_pipeline() -> Pipeline:
    """Construct data preprocessing and estimator pipeline for grid search."""
    numeric_features = ["balance", "income"]
    numeric_transformer = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_features = ["student"]
    categorical_transformer = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
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
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )
    return pipeline


def main(data_path: str = "credit_default_preprocessed.csv") -> dict[str, float]:
    """Perform hyperparameter tuning and log metrics, model, and artifacts to MLflow."""
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

    pipeline = build_pipeline()

    param_grid = {
        "model__C": [0.05, 0.1, 0.5, 1, 2, 5],
        "model__class_weight": ["balanced", None],
        "model__solver": ["liblinear"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("Credit Default Risk - Tuning")

    with mlflow.start_run():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "cv_best_average_precision": float(search.best_score_),
        }

        # Log parameters, metrics, and model to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, best_model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
        )

        # Generate and save confusion matrix artifact
        artifacts_dir = Path("model_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.tight_layout()
        cm_path = artifacts_dir / "tuned_confusion_matrix.png"
        plt.savefig(cm_path, dpi=130)
        plt.close()

        # Save metrics JSON artifact
        metrics_payload = {
            "metrics": metrics,
            "best_params": search.best_params_,
        }
        metrics_json_path = artifacts_dir / "tuned_metrics.json"
        metrics_json_path.write_text(json.dumps(metrics_payload, indent=2))

        # Log file artifacts to MLflow
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_json_path))

        LOG.info("Best parameters: %s", search.best_params_)
        LOG.info("Holdout metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune logistic regression hyperparameters for credit default risk."
    )
    parser.add_argument(
        "--data",
        default="credit_default_preprocessed.csv",
        help="Path to preprocessed CSV dataset.",
    )
    args = parser.parse_args()
    main(args.data)
