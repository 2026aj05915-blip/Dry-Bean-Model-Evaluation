# Dry Bean Classification

A multiclass classification project using UCI's Dry Bean dataset. It compares five classifiers and provides a Streamlit app for evaluating labeled CSV files. This is a learning starter: run the workflow yourself and write your own result analysis before submission.

## Dataset

- Source: UCI Machine Learning Repository, [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry%2Bbean), DOI 10.24432/C50S4B
- 13,611 rows, 16 numeric input features, 7 varieties
- No missing values in the source dataset

## Repository layout

```text
.
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   ├── dry_bean.csv          # full cleaned source data (created by preparation script)
│   └── train_data.csv        # training split (created by preparation script)
├── model/                    # fitted model artifacts and metrics (created by training script)
├── reports/
│   └── screenshots/          # add your BITS Virtual Lab screenshot here
├── src/
│   └── make_split.py         # download, check, and split UCI dataset
├── app.py                    # Streamlit evaluation dashboard
├── build_readme.py           # inserts your generated metrics table into this README
├── requirements.txt
├── smoke_test.py             # checks saved model files and test predictions
├── test_data.csv             # held-out labeled test split (generated)
└── train_models.py           # fits all five models and writes metrics
```

## Run from a terminal

Use a terminal in BITS Virtual Lab and run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python src/make_split.py
python train_models.py
python build_readme.py
python smoke_test.py
streamlit run app.py
```

`src/make_split.py` downloads the data from UCI and creates a stratified 70/30 split. If internet access is unavailable, place UCI's `Dry_Bean_Dataset.xlsx` source file in `data/`; the script will read that local file instead. The held-out test data is written to `test_data.csv`. `train_models.py` fits all five models, saves complete preprocessing/model pipelines in `model/`, writes `model/metrics.csv`, and saves the feature and class metadata. The app loads these saved pipelines and evaluates uploaded data; it does not retrain.

## Models and metrics

The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. The training script reports Accuracy, macro one-vs-rest AUC, macro Precision, macro Recall, macro F1, and MCC.

<!-- RESULTS:START -->
| Model | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9236 | 0.9949 | 0.9359 | 0.9343 | 0.9350 | 0.9078 |
| Decision Tree | 0.9065 | 0.9831 | 0.9183 | 0.9126 | 0.9152 | 0.8869 |
| K-Nearest Neighbors | 0.9243 | 0.9869 | 0.9389 | 0.9349 | 0.9367 | 0.9085 |
| Gaussian Naive Bayes | 0.7676 | 0.9668 | 0.7736 | 0.7704 | 0.7686 | 0.7197 |
| Support Vector Machine | 0.9287 | 0.9956 | 0.9418 | 0.9379 | 0.9397 | 0.9138 |
<!-- RESULTS:END -->

## Your observations

After you run the project, write observations based on your own output:

- Logistic Regression: [write your observation]
- Decision Tree: [write your observation]
- K-Nearest Neighbors: [write your observation]
- Gaussian Naive Bayes: [write your observation]
- Support Vector Machine: [write your observation]
- Overall winner and reason: [write your conclusion]

## App use

Open the deployed app, use the bundled test file or upload a labeled test CSV with all 16 feature columns and a `Class` column, and compare models. The app includes model comparison, diagnostics, confusion matrices, class reports, and downloadable predictions.

## Submission items to complete

- GitHub repository: [add your own link]
- Live Streamlit app: [add your own deployed app link]
- BITS Virtual Lab screenshot: add your own screenshot under `reports/screenshots/`
- Include this README content in your final PDF
