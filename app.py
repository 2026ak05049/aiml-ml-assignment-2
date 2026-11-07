from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

st.set_page_config(page_title="ML Classifier Evaluation", page_icon="📊",
                   layout="wide")
st.title("Machine Learning Classification Dashboard")
st.write(
    "Evaluate five pre-trained classification models using an uploaded "
    "test CSV. The application does not retrain models on uploaded data."
)

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)", type=["csv"],
    help="The CSV must contain all 30 input features and the target column."
)

if uploaded_file is None:
    st.info("Upload the supplied test_data.csv file to begin evaluation.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

model_name = st.selectbox("Choose a classification model", list(MODEL_FILES))
model_path = MODEL_DIR / MODEL_FILES[model_name]

if not model_path.exists():
    st.error(
        f"Saved model not found: {model_path.name}. "
        "Run train_models.py and commit the model folder to GitHub."
    )
    st.stop()

bundle = load_bundle(model_path)
pipeline = bundle["model"]
features = bundle["feature_names"]
target_column = bundle["target_column"]
target_names = bundle["target_names"]

missing = [column for column in features if column not in data.columns]
if missing:
    st.error("The uploaded CSV is missing required feature columns: "
             + ", ".join(missing))
    st.stop()

if target_column not in data.columns:
    st.error(f"The uploaded CSV must include the '{target_column}' target column.")
    st.stop()

X_test = data[features].apply(pd.to_numeric, errors="coerce")
y_test = pd.to_numeric(data[target_column], errors="coerce")

if y_test.isna().any():
    st.error("Target column contains missing or non-numeric values.")
    st.stop()

if X_test.isna().all(axis=1).any():
    st.error("At least one row has no usable feature values.")
    st.stop()

try:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    if len(classes) == 2:
        auc = roc_auc_score(y_test, probabilities[:, 1])
        average = "binary"
    else:
        auc = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="weighted"
        )
        average = "weighted"

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC Score": auc,
        "Precision": precision_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average=average, zero_division=0
        ),
        "MCC Score": matthews_corrcoef(y_test, predictions)
    }

    st.subheader(f"Evaluation results — {model_name}")
    columns = st.columns(3)
    for index, (metric, value) in enumerate(metrics.items()):
        columns[index % 3].metric(metric, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    labels = sorted(pd.Series(y_test).unique().tolist())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(
        [target_names[int(label)] if int(label) < len(target_names)
         else str(label) for label in labels], rotation=20
    )
    ax.set_yticklabels(
        [target_names[int(label)] if int(label) < len(target_names)
         else str(label) for label in labels]
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, predictions, target_names=[
            target_names[int(label)] if int(label) < len(target_names)
            else str(label) for label in labels
        ], zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    with st.expander("Preview uploaded test data"):
        st.dataframe(data.head(20), use_container_width=True)

except Exception as exc:
    st.error(
        "Evaluation could not be completed. Check that target values and "
        f"feature columns match the supplied test CSV. Details: {exc}"
    )
