# Machine Learning Assignment 2 — Classification Models

## a. Problem Statement
The objective is to implement and compare five supervised classification algorithms on one classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). A Streamlit application is developed to evaluate saved models on an uploaded labelled test CSV.

## b. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository; loaded through `sklearn.datasets.load_breast_cancer`.  
**Instances:** 569  
**Input features:** 30 numerical measurements  
**Target:** Two classes — malignant and benign.  
**Problem type:** Binary classification.

The data is split into 80% training and 20% testing using stratified sampling with `random_state=42`. This produced 455 training rows and 114 test rows. The supplied `test_data.csv` contains all 30 input features and the actual target column.

## c. GitHub Repository Link
**Replace before submission:** `[PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]`

## d. Models Used
The following five algorithms are implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Evaluation Metrics

The following values were calculated by running `train_models.py` with the fixed stratified split (random state 42). Precision, recall and F1 use the positive class defined by the dataset (class 1: benign).

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9142 | 0.9429 | 0.9167 | 0.9296 | 0.8139 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance
**Logistic Regression:** Achieved 98.25% accuracy, 0.9954 AUC and 0.9623 MCC. Precision and recall are both 0.9861, indicating a strong and balanced result on this test split.
**Decision Tree:** Achieved 91.23% accuracy and 0.9142 AUC, the lowest among the five models in this experiment. Its results are weaker than the other models on this split; the selected depth and tree structure may limit predictive performance.
**KNN:** Achieved 97.37% accuracy and 1.0000 recall, with 0.9884 AUC. It identifies all positive-class test examples in this split, while its precision of 0.9600 indicates some false-positive predictions.
**Naive Bayes:** Achieved 93.86% accuracy and 0.9878 AUC. Its AUC is relatively high, although its accuracy, F1 and MCC are lower than Logistic Regression, KNN and SVM in this experiment.
**Support Vector Machine:** Achieved 98.25% accuracy, 0.9950 AUC and 0.9623 MCC. Its precision, recall and F1 are balanced and equal to Logistic Regression to four decimal places on this test split.

**Overall winner:** Logistic Regression and Support Vector Machine are joint top performers on this test split: both have 0.9861 F1 and 0.9623 MCC, with Logistic Regression having a marginally higher AUC (0.9954 versus 0.9950). Therefore, Logistic Regression is the overall winner when AUC is used as the tie-breaker. KNN is also competitive and has the highest recall.

## Application
The Streamlit app accepts `test_data.csv`, lets the user select one of the five saved models, and displays the six evaluation metrics, a confusion matrix and a classification report. The saved model is loaded for inference; no training takes place on the uploaded test data.

**Live Streamlit App:** `[PASTE YOUR DEPLOYED STREAMLIT URL HERE]`

## Reproducibility
1. Install dependencies: `pip install -r requirements.txt`
2. Train models and generate test data: `python train_models.py`
3. Launch the app: `streamlit run app.py`
4. Upload `test_data.csv` in the app and select a model.

The training script uses a stratified 80:20 train/test split and random state 42. Preprocessing is stored inside each fitted scikit-learn pipeline to ensure the same transformations are applied during inference.
