"""
Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.
Run this script once before launching the Streamlit app:
    python train_models.py
The script saves five fitted pipelines, a held-out test CSV, and a metrics CSV.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def build_models():
    # Scaling is important for Logistic Regression, KNN and SVM.
    # A single consistent preprocessing pipeline is saved with each model.
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=5, min_samples_split=10, random_state=42
            ))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", C=1.0, probability=True, random_state=42
            ))
        ])
    }

def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_

    if len(classes) == 2:
        auc = roc_auc_score(y_test, probabilities[:, 1])
        average = "binary"
    else:
        auc = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="weighted"
        )
        average = "weighted"

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # The uploaded CSV must contain all features and the actual target.
    test_frame = X_test.copy()
    test_frame["target"] = y_test
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    results = []
    for display_name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "KNN": "knn",
            "Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm"
        }[display_name]

        bundle = {
            "model": estimator,
            "feature_names": list(X.columns),
            "target_names": list(dataset.target_names),
            "target_column": "target",
            "model_name": display_name
        }
        joblib.dump(bundle, MODEL_DIR / f"{safe_name}.joblib")

        row = {"Model": display_name}
        row.update(calculate_metrics(estimator, X_test, y_test))
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT / "model_metrics.csv", index=False)

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository; accessed through scikit-learn",
        "instances": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_names": list(dataset.target_names),
        "test_size": 0.20,
        "random_state": 42
    }
    (ROOT / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nDataset:", metadata["dataset_name"])
    print("Total instances:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Training instances:", X_train.shape[0])
    print("Testing instances:", X_test.shape[0])
    print("\nEvaluation results:")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved models in:", MODEL_DIR)
    print("Saved test_data.csv and model_metrics.csv")

if __name__ == "__main__":
    main()
