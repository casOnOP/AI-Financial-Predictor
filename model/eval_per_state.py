# model/eval_per_state.py
import torch.nn as nn
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODELS_DIR = "model/models"
DATA_DIR = "model/data"
TARGET = "pmjdy_per_1000_pop"
OUT_CSV = "model/per_state_eval.csv"

# Minimal MLP used to load state_dict shape (must match training arch)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def find_latest_pt(models_dir):
    pts = sorted(glob.glob(os.path.join(models_dir, "*.pt")),
                 key=os.path.getmtime)
    return pts[-1] if pts else None


def load_checkpoint_from_pt(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    model_state = ckpt.get("model_state")
    scaler = ckpt.get("scaler")
    imputer = ckpt.get("imputer")
    feature_columns = ckpt.get("feature_columns")
    target_transform_meta = ckpt.get("target_transform_meta", ckpt.get(
        "metadata", {}).get("target_transform", None))
    meta = ckpt.get("metadata", {})
    hidden_dims = tuple(meta.get("hidden_dims", (256, 128)))
    input_dim = len(feature_columns) if feature_columns is not None else None
    output_dim = meta.get("output_dim", 1)
    return {
        "model_state": model_state,
        "scaler": scaler,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "target_transform_meta": target_transform_meta,
        "hidden_dims": hidden_dims,
        "input_dim": input_dim,
        "output_dim": output_dim
    }


class EvalWrapper:
    """Reconstructed wrapper: applies encoders, imputer, scaler, builds MLP and predicts (returns original units)."""

    def __init__(self, model_state, scaler, imputer, feature_columns, hidden_dims, input_dim, output_dim, target_transform_meta, label_encoders):
        self.model_state = model_state
        self.scaler = scaler
        self.imputer = imputer
        self.feature_columns = feature_columns
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_transform_meta = target_transform_meta
        self.label_encoders = label_encoders  # dict col -> encoder or None

    def _build_model(self):
        model = MLP(input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        model.load_state_dict(self.model_state)
        model.eval()
        return model

    def _inverse_target(self, arr):
        tt = self.target_transform_meta or {}
        t = tt.get("type", "none")
        if t == "none":
            return arr
        if t == "log":
            return np.expm1(arr)
        if t == "standardize":
            return arr * tt.get("scale", 1.0) + tt.get("mean", 0.0)
        if t == "robust":
            return arr * tt.get("iqr", 1.0) + tt.get("median", 0.0)
        if t == "winsorize":
            return arr
        return arr

    def predict(self, X_df):
        """X_df: pandas DataFrame with original (raw) columns. Returns numpy preds in original units."""
        # Ensure columns present
        X = X_df.copy()
        # Apply label encoders for categorical columns
        for col, le in self.label_encoders.items():
            if col in X.columns and le is not None:
                X[col] = X[col].astype(str).fillna("__MISSING__")
                X[col] = le.transform(X[col])
        # Reorder to feature_columns
        X_ordered = X[self.feature_columns].values
        # Impute & scale using saved objects
        X_imp = self.imputer.transform(
            X_ordered) if self.imputer is not None else X_ordered
        X_scaled = self.scaler.transform(X_imp).astype(
            "float32") if self.scaler is not None else X_imp.astype("float32")
        xb = torch.from_numpy(X_scaled)
        model = self._build_model()
        with torch.no_grad():
            out = model(xb).cpu().numpy().reshape(-1)
        return self._inverse_target(out)


def build_label_encoders(feature_columns, data_dir):
    """Read all CSVs, detect non-numeric columns among feature_columns, and fit LabelEncoders on combined column values."""
    files = sorted(glob.glob(os.path.join(data_dir, "*_dataset.csv")))
    if not files:
        raise FileNotFoundError("No data files found in " + data_dir)
    # Read all files concatenated but only keep the feature_columns (safe memory: use chunks? but 216k rows OK)
    dfs = []
    for p in files:
        df = pd.read_csv(
            p, usecols=[c for c in feature_columns if c in pd.read_csv(p, nrows=0).columns])
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    encoders = {}
    for col in feature_columns:
        if col not in all_df.columns:
            encoders[col] = None
            continue
        # if dtype is object or contains non-numeric entries -> fit LabelEncoder
        if all_df[col].dtype == object or not pd.api.types.is_numeric_dtype(all_df[col]):
            le = LabelEncoder()
            vals = all_df[col].astype(str).fillna("__MISSING__").values
            le.fit(vals)
            encoders[col] = le
        else:
            encoders[col] = None
    return encoders


def main():
    Path("model").mkdir(exist_ok=True)
    pt_path = find_latest_pt(MODELS_DIR)
    if not pt_path:
        raise SystemExit(
            "No .pt checkpoint found in model/models/ — please ensure training produced a .pt file.")
    print("Loading checkpoint from .pt:", pt_path)
    ck = load_checkpoint_from_pt(pt_path)

    feature_columns = ck["feature_columns"]
    if feature_columns is None:
        raise SystemExit(
            "feature_columns missing in checkpoint; cannot proceed.")

    print("Building label encoders from combined data (this may take a few seconds)...")
    label_encoders = build_label_encoders(feature_columns, DATA_DIR)
    print("Done building encoders.")

    wrapper = EvalWrapper(
        model_state=ck["model_state"],
        scaler=ck["scaler"],
        imputer=ck["imputer"],
        feature_columns=feature_columns,
        hidden_dims=ck.get("hidden_dims", (256, 128)),
        input_dim=ck.get("input_dim", len(feature_columns)),
        output_dim=ck.get("output_dim", 1),
        target_transform_meta=ck.get("target_transform_meta", None),
        label_encoders=label_encoders
    )

    results = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*_dataset.csv"))):
        df = pd.read_csv(f)
        if TARGET not in df.columns:
            print(f"Skipping {f} — target {TARGET} not present")
            continue
        X_df = df.drop(columns=[TARGET])
        y_true = df[TARGET].fillna(0).values.astype(float)
        # Predict
        preds = wrapper.predict(X_df)
        preds = np.asarray(preds, dtype=float).reshape(-1)
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        results.append({"file": os.path.basename(f), "n": len(
            y_true), "rmse": rmse, "mae": mae, "r2": r2})
        print(
            f"{os.path.basename(f)}  rows={len(y_true)}  RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.4f}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)


if __name__ == "__main__":
    main()
