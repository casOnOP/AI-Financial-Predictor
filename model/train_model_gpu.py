import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "population_density",
    "literacy_rate",
    "female_ratio",
    "agri_dependence",
    "avg_income",
    "employment_rate",
    "bank_branches",
    "financial_awareness_index",
    "avg_savings_rate",
    "main_crop",  # label-encoded
]

TARGET_COL = "recommended_scheme"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "district" not in df.columns:
        df["district"] = np.arange(len(df)).astype(str)

    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    numeric_cols = [c for c in FEATURE_COLS if c != "main_crop"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["main_crop"] = df["main_crop"].astype(str)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


def preprocess_labels(df: pd.DataFrame):
    encoders = {}

    le_crop = LabelEncoder()
    df["main_crop"] = le_crop.fit_transform(df["main_crop"].astype(str))
    encoders["main_crop"] = le_crop

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET_COL].astype(str))
    encoders["target"] = le_target

    return df, y, encoders


def train_xgb(X: np.ndarray, y: np.ndarray, epochs: int, max_depth: int, lr: float):
    model = xgb.XGBClassifier(
        n_estimators=epochs,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",  # ✅ GPU mode
        device="cuda",       # ✅ Enables CUDA GPU
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        random_state=42,
    )
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train GPU XGBoost Financial Model")
    parser.add_argument("--data", default="data/karnataka_dataset.csv",
                        help="Training CSV path")
    parser.add_argument("--models_dir", default="models",
                        help="Where to save models")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of boosting rounds (epochs)")
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.08)

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    df = load_data(args.data)
    df, y, enc = preprocess_labels(df)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    print("\n⚠️  Tiny dataset detected — using full dataset for training (no validation split)\n")

    print(f"🚀 Training GPU XGBoost model with {args.epochs} epochs...")
    model = train_xgb(X, y, args.epochs, args.max_depth, args.learning_rate)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")

    print(f"\n✅ Training Completed")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print("\n📊 Per-class Report:")
    print(classification_report(y, y_pred, digits=4,
                                target_names=enc["target"].classes_))

    model_path = os.path.join(args.models_dir, "financial_model_gpu.xgb")
    model.get_booster().save_model(model_path)

    with open(os.path.join(args.models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(enc, f)

    feat_path = os.path.join(args.models_dir, "feature_columns.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"\n💾 Saved model to: {model_path}")
    print(
        f"💾 Saved encoders to: {os.path.join(args.models_dir, 'label_encoder.pkl')}")
    print(f"💾 Saved feature list to: {feat_path}\n")


if __name__ == "__main__":
    main()
