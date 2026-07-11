import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

REQUIRED_COLUMNS = [
    "district",
    "population_density",
    "literacy_rate",
    "female_ratio",
    "agri_dependence",
    "main_crop",
    "avg_income",
    "employment_rate",
    "bank_branches",
    "financial_awareness_index",
    "avg_savings_rate",
    "recommended_scheme",
]

CATEGORICAL_COLS = ["main_crop", "recommended_scheme"]


def read_all_csvs(data_dir: str) -> pd.DataFrame:
    files = [f for f in glob(os.path.join(data_dir, "*.csv"))]
    # Avoid merging the output back into itself on re-runs
    files = [f for f in files if not os.path.basename(f).startswith("merged_")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, comment="#")
        frames.append(df)
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return merged


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only required columns; if missing, create
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS].copy()

    # Dtypes
    float_cols = [
        "population_density",
        "literacy_rate",
        "female_ratio",
        "agri_dependence",
        "avg_income",
        "employment_rate",
        "financial_awareness_index",
        "avg_savings_rate",
    ]
    int_cols = ["bank_branches"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Clean strings
    df["district"] = df["district"].astype(str).str.strip()
    df["main_crop"] = df["main_crop"].astype(str).str.strip().str.lower()
    df["recommended_scheme"] = df["recommended_scheme"].astype(
        str).str.strip().str.upper()

    # Drop duplicate district rows, keep last occurrence
    df = df.drop_duplicates(subset=["district"], keep="last")

    # Basic sanity caps
    df["literacy_rate"] = df["literacy_rate"].clip(0, 1)
    df["female_ratio"] = df["female_ratio"].clip(0.8, 1.2)  # ratio (F/M)
    df["agri_dependence"] = df["agri_dependence"].clip(0, 1)
    df["employment_rate"] = df["employment_rate"].clip(0, 1)
    df["financial_awareness_index"] = df["financial_awareness_index"].clip(
        0, 1)
    df["avg_savings_rate"] = df["avg_savings_rate"].clip(0, 1)

    return df


def label_encode(df: pd.DataFrame):
    encoders = {}
    df_out = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_out[col] = le.fit_transform(df_out[col].astype(str))
        encoders[col] = {
            "classes": le.classes_.tolist(),
        }
    return df_out, encoders


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSVs and label-encode categorical columns.")
    parser.add_argument("--data_dir", default="model/data",
                        help="Directory with input CSV files")
    parser.add_argument(
        "--out_csv", default="model/data/merged_training.csv", help="Output merged CSV path")
    parser.add_argument("--enc_json", default="model/models/merge_label_encoders.json",
                        help="Where to save encoder metadata")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.enc_json), exist_ok=True)

    merged = read_all_csvs(args.data_dir)
    merged = coerce_schema(merged)
    encoded, encoders = label_encode(merged)

    encoded.to_csv(args.out_csv, index=False)
    with open(args.enc_json, "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2)

    print(f"Saved merged CSV → {args.out_csv}")
    print(f"Saved encoder metadata → {args.enc_json}")


if __name__ == "__main__":
    main()
