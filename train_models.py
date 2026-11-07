"""Train five classifiers and save metrics and model files."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(ROOT / "data" / "train_data.csv")
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"].astype(str)
test_df = pd.read_csv(ROOT / "test_data.csv")
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"].astype(str)
feature_names = list(X_train.columns)
classes = sorted(y_train.unique())

# Scaling is important for LR, KNN, and SVM. Tree and Gaussian NB use imputation only.
def scaled_model(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])

def unscaled_model(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])

models = {
    "Logistic Regression": scaled_model(LogisticRegression(max_iter=3000)),
    "Decision Tree": unscaled_model(DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=5465)),
    "K-Nearest Neighbors": scaled_model(KNeighborsClassifier(n_neighbors=7, weights="distance")),
    "Gaussian Naive Bayes": unscaled_model(GaussianNB()),
    "Support Vector Machine": scaled_model(SVC(kernel="rbf", probability=True, random_state=5465)),
}

rows = []
for display_name, model in models.items():
    print(f"Training {display_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    model_classes = model.named_steps["classifier"].classes_
    # Multiclass one-vs-rest ROC AUC; macro gives every class equal weight.
    auc = roc_auc_score(y_test, probabilities, labels=model_classes,
                        multi_class="ovr", average="macro")
    rows.append({
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC_macro_ovr": auc,
        "Precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    })
    filename = display_name.lower().replace(" ", "_").replace("-", "") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

metrics = pd.DataFrame(rows)
metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
(ROOT / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
(ROOT / "class_names.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
print("\nMetrics from this run:")
print(metrics.round(4).to_string(index=False))
print("\nSaved models to model/ and metrics to model/metrics.csv")
