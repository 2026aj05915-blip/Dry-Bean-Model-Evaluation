"""Download and prepare UCI Dry Bean data, then create a stratified holdout split."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

local_file = DATA_DIR / "Dry_Bean_Dataset.xlsx"
if local_file.exists():
    print(f"Reading the uploaded source file: {local_file}")
    data = pd.read_excel(local_file)
    if "Class" not in data.columns:
        raise ValueError("The Excel source file must contain the `Class` target column.")
    data["Class"] = data["Class"].astype(str)
else:
    print("Downloading Dry Bean dataset from UCI (dataset id 602)...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # UCI supplies one target column called Class. Keep it out of the input features.
    if y.shape[1] != 1:
        raise ValueError(f"Expected one target column, found: {list(y.columns)}")
    target_name = y.columns[0]
    data = X.copy()
    data["Class"] = y[target_name].astype(str).to_numpy()

# Basic checks before making the split.
if data.isna().any().any():
    raise ValueError("Dataset contains missing values; inspect before proceeding.")
if data["Class"].nunique() < 2:
    raise ValueError("Classification needs at least two target classes.")

train_df, test_df = train_test_split(
    data,
    test_size=0.30,
    random_state=5465,  # Change this to a seed you choose and document.
    stratify=data["Class"],
)

# Save raw cleaned data and a held-out labeled test CSV for the app.
data.to_csv(DATA_DIR / "dry_bean.csv", index=False)
train_df.to_csv(DATA_DIR / "train_data.csv", index=False)
test_df.to_csv(ROOT / "test_data.csv", index=False)

print(f"Rows: {len(data):,}; input features: {data.drop(columns=['Class']).shape[1]}; classes: {data['Class'].nunique()}")
print(f"Training rows: {len(train_df):,}; test rows: {len(test_df):,}")
print("Class counts:")
print(data["Class"].value_counts().sort_index().to_string())
print("\nCreated data/dry_bean.csv, data/train_data.csv, and test_data.csv")

