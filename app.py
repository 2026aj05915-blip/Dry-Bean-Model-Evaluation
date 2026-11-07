"""Polished evaluation dashboard for the saved Dry Bean classifiers.

The app only loads saved models and evaluates labeled CSV data. It never trains.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Dry Bean | Model Evaluation Studio",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px;
             background: linear-gradient(120deg, #12372a 0%, #1e5a42 55%, #d9a441 150%);
             color: white; margin-bottom: 1.2rem;}
      .hero h1 {color: white; margin: 0 0 .25rem 0;}
      .hero p {color: #e7f0e9; margin: 0; font-size: 1.02rem;}
      .small-note {color: #667085; font-size: .88rem;}
      div[data-testid="stMetric"] {background: #f6f8f6; border: 1px solid #e4ebe5;
        padding: .85rem 1rem; border-radius: 14px;}
      div[data-testid="stMetricLabel"] {font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def read_metadata():
    feature_file = ROOT / "feature_names.json"
    class_file = ROOT / "class_names.json"
    if not feature_file.exists() or not class_file.exists():
        raise FileNotFoundError("Metadata is missing. Run make_split.py and train_models.py first.")
    return (
        json.loads(feature_file.read_text(encoding="utf-8")),
        json.loads(class_file.read_text(encoding="utf-8")),
    )


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model is missing: {model_path.name}. Run train_models.py first.")
    return joblib.load(model_path)


def score_model(model, frame, features, classes):
    """Return metrics and predictions for one saved pipeline."""
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    y_true = frame["Class"].astype(str).str.strip()
    y_pred = model.predict(X)
    probabilities = model.predict_proba(X)
    fitted_classes = list(model.named_steps["classifier"].classes_)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC (macro OVR)": float("nan"),
        "Precision (macro)": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # Multiclass OVR AUC needs every expected class represented in y_true.
    if set(classes).issubset(set(y_true.unique())):
        metrics["AUC (macro OVR)"] = roc_auc_score(
            y_true,
            probabilities,
            labels=fitted_classes,
            multi_class="ovr",
            average="macro",
        )
    return metrics, y_true, y_pred, probabilities


try:
    FEATURES, CLASSES = read_metadata()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
    <div class="hero">
      <h1>Dry Bean · Model Evaluation Studio</h1>
      <p>Compare five saved classifiers on the same labeled test file, inspect their errors, and export predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Evaluation setup")
    st.write("Choose the held-out example file or upload another labeled CSV.")
    bundled_path = ROOT / "test_data.csv"
    use_bundled = st.checkbox("Use the project test file", value=bundled_path.exists(), disabled=not bundled_path.exists())
    uploaded = st.file_uploader("Or upload a labeled CSV", type=["csv"], help="Must include the 16 input columns and a `Class` target column.")
    st.divider()
    st.caption("Target: `Class` · 7 bean varieties")
    st.caption("The app loads saved models; it never retrains on uploaded data.")

frame = None
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        input_label = uploaded.name
    except Exception as exc:
        st.error(f"I could not read this CSV: {exc}")
        st.stop()
elif use_bundled and bundled_path.exists():
    try:
        frame = pd.read_csv(bundled_path)
        input_label = "Bundled held-out test data"
    except Exception as exc:
        st.error(f"I could not read test_data.csv: {exc}")
        st.stop()

if frame is None:
    st.info("Choose **Use the project test file** or upload a labeled CSV in the sidebar to begin.")
    st.markdown("#### What this app evaluates")
    intro_cols = st.columns(3)
    intro_cols[0].metric("Input features", len(FEATURES))
    intro_cols[1].metric("Bean varieties", len(CLASSES))
    intro_cols[2].metric("Saved classifiers", len(MODEL_FILES))
    st.stop()

missing = [column for column in FEATURES + ["Class"] if column not in frame.columns]
if missing:
    st.error("This file is missing required columns: " + ", ".join(missing))
    st.stop()
if frame.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

frame = frame.copy()
frame["Class"] = frame["Class"].astype(str).str.strip()
unknown = sorted(set(frame["Class"].unique()) - set(CLASSES))
if unknown:
    st.error("Unknown class labels found: " + ", ".join(unknown))
    st.stop()

# Convert feature columns to numeric. The fitted median imputer handles individual blank cells.
for column in FEATURES:
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
fully_empty = [column for column in FEATURES if frame[column].isna().all()]
if fully_empty:
    st.error("These feature columns contain no usable numbers: " + ", ".join(fully_empty))
    st.stop()

if len(frame) < 20:
    st.warning("This is a small test file. Metrics may vary substantially with so few rows.")
extra_columns = [column for column in frame.columns if column not in FEATURES + ["Class"]]
if extra_columns:
    st.caption("Extra columns will be ignored: " + ", ".join(extra_columns))
missing_values = int(frame[FEATURES].isna().sum().sum())
if missing_values:
    st.info(f"Found {missing_values} blank or non-numeric feature values. The saved pipelines will impute them using training medians.")

st.caption(f"Evaluating **{len(frame):,} rows** from **{input_label}**")

with st.spinner("Scoring the same test rows with all five saved models…"):
    all_results = {}
    for name in MODEL_FILES:
        try:
            model = load_model(name)
            all_results[name] = score_model(model, frame, FEATURES, CLASSES)
        except Exception as exc:
            st.error(f"Could not score {name}: {exc}")
            st.stop()

comparison = pd.DataFrame({name: result[0] for name, result in all_results.items()}).T
comparison.index.name = "Model"
comparison = comparison.reset_index()
comparison = comparison[["Model", "Accuracy", "AUC (macro OVR)", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC"]]

if comparison["F1 (macro)"].notna().any():
    best_row = comparison.loc[comparison["F1 (macro)"].idxmax()]
    st.success(f"Highest macro-F1 on this file: **{best_row['Model']}** ({best_row['F1 (macro)']:.3f}). Metrics describe this uploaded test file only.")

tab_compare, tab_detail, tab_predictions, tab_about = st.tabs(
    ["Model comparison", "Model diagnostics", "Predictions", "About this evaluation"]
)

with tab_compare:
    st.subheader("All five models on the same test rows")
    st.caption("Macro metrics give each bean variety equal weight. AUC is one-vs-rest. MCC is multiclass-aware.")
    st.dataframe(
        comparison.style.format({column: "{:.4f}" for column in comparison.columns if column != "Model"})
        .background_gradient(subset=["Accuracy", "AUC (macro OVR)", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC"], cmap="YlGn"),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download comparison as CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="dry_bean_model_comparison.csv",
        mime="text/csv",
    )

with tab_detail:
    selected = st.selectbox("Choose a model for detailed diagnostics", list(MODEL_FILES), key="detail_model")
    metrics, y_true, y_pred, _ = all_results[selected]
    st.subheader(selected)
    metric_columns = st.columns(6)
    for column, metric_name in zip(
        metric_columns,
        ["Accuracy", "AUC (macro OVR)", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC"],
    ):
        value = metrics[metric_name]
        column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.3f}")
    if pd.isna(metrics["AUC (macro OVR)"]):
        st.info("AUC is unavailable because this uploaded file does not contain all seven classes.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Confusion matrix")
        st.caption("Rows are actual varieties; columns are predicted varieties.")
        matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
        matrix_df = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
        st.dataframe(matrix_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
    with right:
        st.markdown("#### Per-class report")
        report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

with tab_predictions:
    selected = st.selectbox("Choose a model for row-level predictions", list(MODEL_FILES), key="prediction_model")
    metrics, y_true, y_pred, probabilities = all_results[selected]
    output = frame.copy()
    output["Actual_Class"] = y_true.to_numpy()
    output["Predicted_Class"] = y_pred
    output["Correct"] = output["Actual_Class"] == output["Predicted_Class"]
    output["Confidence"] = probabilities.max(axis=1)
    st.caption("Confidence is the selected model’s highest predicted class probability for each row.")
    show_errors = st.checkbox("Show only incorrect predictions", value=False)
    visible = output.loc[~output["Correct"]] if show_errors else output
    st.dataframe(visible, hide_index=True, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=f"dry_bean_predictions_{selected.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with tab_about:
    st.subheader("How to read the metrics")
    st.markdown(
        """
        - **Accuracy**: fraction of all rows classified correctly.
        - **AUC (macro OVR)**: average one-vs-rest class ranking quality; requires every class in the labeled file.
        - **Precision / Recall / F1 (macro)**: calculated per variety, then averaged equally across varieties.
        - **MCC**: a balanced summary of classification quality that supports multiple classes.

        Each classifier is loaded from `model/`. Preprocessing is part of the saved pipeline. Uploaded CSVs are used only for evaluation and prediction; the app never fits a model on them.
        """
    )
    st.markdown("#### Required columns")
    with st.expander("Show the 16 feature columns and target"):
        st.code("\n".join(FEATURES + ["Class"]))
    st.markdown("Dataset: [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2Bbean).")

st.divider()
st.markdown('<p class="small-note">Dry Bean Classification · Model evaluation only · Built for the ML Assignment</p>', unsafe_allow_html=True)
