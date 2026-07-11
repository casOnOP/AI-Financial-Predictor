import os
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

# -------------------------
# Model & wrapper
# -------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InferenceWrapper:
    """Pickle-friendly wrapper exposing predict(X) and handling inverse target transform."""

    def __init__(self, model_state_dict, scaler, imputer, feature_columns, hidden_dims, input_dim, output_dim, task, label_classes, target_transform_meta):
        self.model_state_dict = model_state_dict
        self.scaler = scaler
        self.imputer = imputer
        self.feature_columns = feature_columns
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.label_classes = label_classes
        self.target_transform_meta = target_transform_meta

    def _build_model(self):
        model = MLP(input_dim=self.input_dim,
                    hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        model.load_state_dict(self.model_state_dict)
        model.eval()
        return model

    def _inverse_target(self, arr):
        # arr is numpy array in transformed scale
        tt = self.target_transform_meta
        if tt is None or tt.get("type", "none") == "none":
            return arr
        t = tt["type"]
        if t == "log":
            return np.expm1(arr)
        if t == "standardize":
            mu = tt["mean"]
            sigma = tt["scale"]
            return arr * sigma + mu
        if t == "robust":
            med = tt["median"]
            iqr = tt["iqr"]
            return arr * iqr + med
        if t == "winsorize":
            # winsorize stored min/max used; we assume arr is already within bounds so just return arr
            return arr
        return arr

    def predict(self, X):
        import numpy as np
        import torch
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        X_imp = self.imputer.transform(X_np)
        X_scaled = self.scaler.transform(X_imp).astype("float32")
        xb = torch.from_numpy(X_scaled)
        model = self._build_model()
        with torch.no_grad():
            out = model(xb).cpu().numpy().reshape(-1)
        # inverse transform target
        preds_orig = self._inverse_target(out)
        if self.task == "classification" and self.label_classes is not None:
            # classification case (not covered here)
            return preds_orig
        return preds_orig

# -------------------------
# Data prep & transforms
# -------------------------


def load_and_prepare(data_dir, target_name=None, target_transform="log", winsor_limits=(0.01, 0.99), val_size=0.15, seed=42):
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*_dataset.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No '*_dataset.csv' files found in {data_dir}")
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)
    if target_name is None:
        target_name = data.columns[-1]
    X = data.drop(columns=[target_name])
    y = data[target_name].fillna(0).astype(float)

    # Drop rows where all features are NaN? (simple check)
    # Encode categorical feature columns as numbers
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Impute numerical features
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Target transform
    tmeta = {"type": "none"}
    y_trans = y.copy()
    if target_transform == "none":
        pass
    elif target_transform == "log":
        # use log1p to handle zeros
        y_trans = np.log1p(y)
        tmeta = {"type": "log"}
    elif target_transform == "standardize":
        mu = y.mean()
        sigma = y.std() if y.std() != 0 else 1.0
        y_trans = (y - mu) / sigma
        tmeta = {"type": "standardize",
                 "mean": float(mu), "scale": float(sigma)}
    elif target_transform == "robust":
        med = np.median(y)
        iqr = np.subtract(*np.percentile(y, [75, 25])) or 1.0
        y_trans = (y - med) / iqr
        tmeta = {"type": "robust", "median": float(med), "iqr": float(iqr)}
    elif target_transform == "winsorize":
        lowp, highp = winsor_limits
        lo = np.percentile(y, lowp*100)
        hi = np.percentile(y, highp*100)
        y_trans = y.clip(lo, hi)
        tmeta = {"type": "winsorize", "low": float(lo), "high": float(hi)}
    else:
        raise ValueError("unsupported target_transform")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_trans, test_size=val_size, random_state=seed, shuffle=True)
    return {
        "X_train": X_train.astype("float32"),
        "X_val": X_val.astype("float32"),
        "y_train": np.asarray(y_train, dtype="float32"),
        "y_val": np.asarray(y_val, dtype="float32"),
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
        "task": "regression",
        "output_dim": 1,
        "target_transform_meta": tmeta,
        "target_original_sample": {"mean": float(y.mean()), "std": float(y.std()), "median": float(np.median(y))}
    }


def inverse_transform_preds(preds_trans, tmeta):
    # preds_trans is numpy array in transformed space
    if tmeta is None or tmeta.get("type", "none") == "none":
        return preds_trans
    t = tmeta["type"]
    if t == "log":
        return np.expm1(preds_trans)
    if t == "standardize":
        return preds_trans * tmeta["scale"] + tmeta["mean"]
    if t == "robust":
        return preds_trans * tmeta["iqr"] + tmeta["median"]
    if t == "winsorize":
        # preds already in original units
        return preds_trans
    return preds_trans

# -------------------------
# Training & eval
# -------------------------


def build_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = TensorDataset(torch.from_numpy(
        X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb).view(-1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / (n or 1)


def eval_epoch(model, device, loader, task, tmeta):
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb).view(-1).cpu().numpy()
            ys.append(yb.cpu().numpy())
            preds.append(out)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    # invert transform to original units
    y_orig = inverse_transform_preds(ys, tmeta)
    preds_orig = inverse_transform_preds(preds, tmeta)
    mse = mean_squared_error(y_orig, preds_orig)
    mae = mean_absolute_error(y_orig, preds_orig)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_orig, preds_orig)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="model/data")
    parser.add_argument("--target", default=None)
    parser.add_argument("--target-transform", default="log",
                        choices=["none", "log", "standardize", "robust", "winsorize"])
    parser.add_argument("--winsor-limits", default="0.01,0.99",
                        help="low,high quantiles for winsorize")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--output-dir", default="model/models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    wlow, whigh = map(float, args.winsor_limits.split(","))
    prepared = load_and_prepare(args.data_dir, target_name=args.target, target_transform=args.target_transform, winsor_limits=(
        wlow, whigh), val_size=0.15, seed=args.seed)

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    y_train = prepared["y_train"]
    y_val = prepared["y_val"]
    imputer = prepared["imputer"]
    scaler = prepared["scaler"]
    feature_columns = prepared["feature_columns"]
    tmeta = prepared["target_transform_meta"]
    task = prepared["task"]
    output_dim = prepared["output_dim"]
    input_dim = X_train.shape[1]
    hidden_dims = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims,
                output_dim=output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    train_loader, val_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, args.batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_rmse = float("inf")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pt_name = f"financial_model_{timestamp}.pt"
    meta = {"created_at": timestamp, "target_transform": tmeta, "hidden_dims": hidden_dims,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}

    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(
            model, device, train_loader, optimizer, criterion)
        val_metrics = eval_epoch(model, device, val_loader, task, tmeta)
        print(
            f"Epoch {epoch}/{args.epochs}  TrainLoss(transformed): {train_loss:.6f}  ValRMSE: {val_metrics['rmse']:.3f}  ValMAE: {val_metrics['mae']:.3f}  ValR2: {val_metrics['r2']:.4f}")
        # Save best by RMSE on original scale
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            ckpt = {"model_state": model.state_dict(), "scaler": scaler, "imputer": imputer,
                    "feature_columns": feature_columns, "task": task, "target_transform_meta": tmeta, "metadata": meta}
            pt_path = os.path.join(args.output_dir, pt_name)
            torch.save(ckpt, pt_path)
            print("Saved best .pt ->", pt_path)

    # Save pickled wrapper
    wrapper = InferenceWrapper(model_state_dict=model.state_dict(), scaler=scaler, imputer=imputer, feature_columns=feature_columns,
                               hidden_dims=hidden_dims, input_dim=input_dim, output_dim=output_dim, task=task, label_classes=None, target_transform_meta=tmeta)
    pkl_path = os.path.join(args.output_dir, "trained_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(wrapper, f)
    # Save metadata file
    with open(os.path.join(args.output_dir, f"metadata_{timestamp}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # Save target transform metadata separately for easy inspection
    with open(os.path.join(args.output_dir, f"target_transform_{timestamp}.json"), "w") as f:
        json.dump(tmeta, f, indent=2)

    print("Training complete. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
