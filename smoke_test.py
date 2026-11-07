"""Check that saved classifiers can score the held-out test file."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
features = json.loads((ROOT / "feature_names.json").read_text(encoding="utf-8"))
test = pd.read_csv(ROOT / "test_data.csv")
metrics = pd.read_csv(ROOT / "model" / "metrics.csv").set_index("Model")
X, y = test[features], test["Class"].astype(str)

for name, filename in MODELS.items():
    path = ROOT / "model" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    model = joblib.load(path)
    predictions = model.predict(X)
    score = accuracy_score(y, predictions)
    recorded = float(metrics.loc[name, "Accuracy"])
    if abs(score - recorded) > 1e-12:
        raise AssertionError(f"{name}: saved model accuracy {score} differs from metrics.csv {recorded}")
    print(f"OK: {name} loaded and scored {len(test)} rows (accuracy={score:.4f})")

print("All saved models passed the smoke check.")
