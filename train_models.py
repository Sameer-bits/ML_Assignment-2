"""
Training script for the Dry Bean classification ML assignment.

Run from the project root:
    python train_models.py

What it does:
1. Loads the original Excel or CSV dataset.
2. Removes duplicate rows and validates the data.
3. Creates a stratified train/test split.
4. Trains Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and SVM.
5. Evaluates each model using Accuracy, AUC, Precision,
   Recall, F1 and multiclass MCC.
6. Saves fitted models and evaluation outputs.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Resolve paths 
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

TARGET = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]


def load_dataset() -> pd.DataFrame:
    """Load the original Dry Bean Excel file, or its CSV version."""
    excel_path = DATA_DIR / "Dry_Bean_Dataset.xlsx"
    csv_path = DATA_DIR / "Dry_Bean_Dataset.csv"

    if excel_path.exists():
        data = pd.read_excel(excel_path)
        print(f"Loaded: {excel_path}")
    elif csv_path.exists():
        data = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path}")
    else:
        raise FileNotFoundError(
            "Dataset not found. Put Dry_Bean_Dataset.xlsx or "
            "Dry_Bean_Dataset.csv in the root folder."
        )
    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, remove duplicate rows and reject missing/non-numeric data."""
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned = data[required].copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(cleaned)} duplicate rows.")

    if cleaned.isnull().any().any():
        null_counts = cleaned.isnull().sum()
        raise ValueError(f"Missing values found:\n{null_counts[null_counts > 0]}")

    for col in FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if cleaned[FEATURES].isnull().any().any():
        raise ValueError("One or more feature columns contain non-numeric values.")

    if not np.isfinite(cleaned[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite or non-finite values.")

    if cleaned[TARGET].nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    return cleaned


def split_dataset(data: pd.DataFrame):
    """Create a reproducible, stratified 80/20 train/test split."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_frame = X_train.copy()
    train_frame[TARGET] = y_train
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(DATA_DIR / "train_data.csv", index=False)
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """Return all five model definitions."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def calculate_metrics(model, X_test, y_test) -> dict:
    """Calculate the six required metrics for one fitted classifier."""
    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    # AUC is calculated as macro one-vs-rest for this multiclass problem.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        try:
            results["AUC"] = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=model.classes_,
            )
        except ValueError:
            results["AUC"] = np.nan
    else:
        results["AUC"] = np.nan

    return results


def train_and_evaluate(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit every model, save it, and write comparison/evaluation files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    class_labels = sorted(pd.Series(y_test).unique())

    for display_name, model in build_models().items():
        print(f"\nTraining {display_name} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            model.fit(X_train, y_train)

        filename_map = {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "KNN": "knn.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "SVM": "svm.joblib",
        }
        model_path = MODEL_DIR / filename_map[display_name]
        joblib.dump(model, model_path)

        metrics = calculate_metrics(model, X_test, y_test)
        summary_rows.append({"Model": display_name, **metrics})

        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=class_labels)
        safe_name = display_name.lower().replace(" ", "_")

        pd.DataFrame(
            cm, index=class_labels, columns=class_labels
        ).to_csv(REPORT_DIR / f"{safe_name}_confusion_matrix.csv")

        report = classification_report(
            y_test, predictions, labels=class_labels,
            output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(
            REPORT_DIR / f"{safe_name}_classification_report.csv"
        )

        print(f"Saved model: {model_path}")
        print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

    results = pd.DataFrame(summary_rows)
    results = results[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    print("\nFinal model comparison:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return results


def main():
    """Run the complete end-to-end workflow."""
    raw_data = load_dataset()
    cleaned_data = clean_dataset(raw_data)

    cleaned_path = ROOT / "Dry_Bean_Cleaned.csv"
    cleaned_data.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved: {cleaned_path}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    X_train, X_test, y_train, y_test = split_dataset(cleaned_data)
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
