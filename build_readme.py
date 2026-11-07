"""Update the README results section from model/metrics.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
readme_path = ROOT / "README.md"
metrics_path = ROOT / "model" / "metrics.csv"
start_marker = "<!-- RESULTS:START -->"
end_marker = "<!-- RESULTS:END -->"

if not metrics_path.exists():
    raise FileNotFoundError("Metrics are missing. Run `python train_models.py` first.")
text = readme_path.read_text(encoding="utf-8-sig")
if start_marker not in text or end_marker not in text:
    raise ValueError("README results markers are missing; restore the markers before rebuilding.")
metrics = pd.read_csv(metrics_path)
header = "| Model | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |\n|---|---:|---:|---:|---:|---:|---:|"
rows = []
for _, row in metrics.iterrows():
    rows.append("| " + str(row["Model"]) + " | " + " | ".join(f"{row[column]:.4f}" for column in ["Accuracy", "AUC_macro_ovr", "Precision_macro", "Recall_macro", "F1_macro", "MCC"]) + " |")
replacement = start_marker + "\n" + header + "\n" + "\n".join(rows) + "\n" + end_marker
text = text[:text.index(start_marker)] + replacement + text[text.index(end_marker) + len(end_marker):]
readme_path.write_text(text, encoding="utf-8")
print("README metric table updated from your model/metrics.csv")
