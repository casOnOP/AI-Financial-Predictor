import os
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

# -------------------------
# Model / wrapper classes
# -------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), output_dim=1, task="regression", dropout=0.1):
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
    """
    Pickle-friendly wrapper exposing .predict(X) (numpy array or 2D list).
    This wrapper holds: model_state_dict, scaler, imputer, feature_columns, task, label_classes, hidden_dims.
    On predict it reconstructs the MLP and runs inference on CPU.
    """

    def __init__(self, model_state_dict, scaler, imputer, feature_columns, task, label_classes, hidden_dims, input_dim, output_dim):
        self.model_state_dict = model_state_dict
        self.scaler = scaler
        self.imputer = imputer
        self.feature_columns = feature_columns
        self.task = task
        self.label_classes = label_classes
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # don't instantiate large torch model here (will instantiate inside predict on CPU)

    def _build_model(self):
        model = MLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims,
                    output_dim=self.output_dim, task=self.task)
        model.load_state_dict(self.model_state_dict)
        model.eval()
        return model

    def predict(self, X):
        """
        X: numpy array shaped (n_samples, n_features) or list of lists
        Returns: numpy array of predictions (for classification returns original labels if label_classes present)
        """
        import numpy as np
        import torch

        X_np = np.asarray(X, dtype=float)
        # If a single row (1D), coerce to 2D
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        # Impute & scale
        X_imp = self.imputer.transform(X_np)
        X_scaled = self.scaler.transform(X_imp).astype("float32")
        xb = torch.from_numpy(X_scaled)
        model = self._build_model()
        with torch.no_grad():
            out = model(xb)
            if self.task == "classification":
                preds = out.argmax(dim=1).cpu().numpy()
                if self.label_classes is not None:
                    return np.array([self.label_classes[int(p)] for p in preds])
                return preds
            else:
                return out.view(-1).cpu().numpy()

# -------------------------
# Data prep & training util
# -------------------------


def load_and_prepare(data_dir, target_name=None, test_size=0.15, random_state=42):
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*_dataset.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No '*_dataset.csv' files found in {data_dir}")

    print(f"Found {len(csv_paths)} csv files. Loading and concatenating...")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df)
    data = pd.concat(dfs, axis=0, ignore_index=True)
    print(
        f"Total rows after concat: {len(data)}; columns: {list(data.columns)}")

    # Determine target
    if target_name is None:
        target_name = data.columns[-1]  # fallback to last column
        print(
            f"No --target provided; using last column as target: '{target_name}'")
    elif target_name not in data.columns:
        print(
            f"Warning: provided target '{target_name}' not found in columns. Falling back to last column '{data.columns[-1]}'")
        target_name = data.columns[-1]

    # Separate X/y
    X = data.drop(columns=[target_name])
    y = data[target_name]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(
        f"Numeric columns: {len(num_cols)}, Categorical columns: {len(cat_cols)}")

    # Encode categorical columns with LabelEncoder (per-column)
    label_encoders = {}
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # not currently saved but kept if needed

    # Impute & scale
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Determine task & prepare target
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 20:
            task = "regression"
            y_processed = y.astype(float).values
            output_dim = 1
            print("Heuristic: numeric target with many unique values -> regression")
            label_encoder_y = None
        else:
            task = "classification"
            le_y = LabelEncoder()
            y_processed = le_y.fit_transform(
                y.astype(str).fillna("__MISSING_Y__"))
            output_dim = int(y_processed.max()) + 1
            label_encoder_y = le_y
            print(
                f"Numeric target with few uniques -> classification with {output_dim} classes")
    else:
        task = "classification"
        le_y = LabelEncoder()
        y_processed = le_y.fit_transform(y.astype(str).fillna("__MISSING_Y__"))
        output_dim = int(y_processed.max()) + 1
        label_encoder_y = le_y
        print(
            f"Categorical target -> classification with {output_dim} classes")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_processed, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}")

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder_y": label_encoder_y,
        "feature_columns": X.columns.tolist(),
        "task": task,
        "output_dim": output_dim
    }


def build_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    # Ensure y arrays are numpy arrays
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    train_ds = TensorDataset(torch.from_numpy(
        X_train), torch.from_numpy(y_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(
        X_val), torch.from_numpy(y_val.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_loop(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(out, yb.long())
        else:
            out = out.view(-1)
            loss = criterion(out, yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n if n > 0 else 0.0


def eval_loop(model, device, val_loader, criterion, task):
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            if task == "classification":
                preds = out.argmax(dim=1)
                correct += (preds == yb.long()).sum().item()
                total += xb.size(0)
                loss = criterion(out, yb.long())
            else:
                out = out.view(-1)
                loss = criterion(out, yb.float())
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc = (correct / total) if total > 0 else None
    return avg_loss, acc

# -------------------------
# Entrypoint
# -------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="model/data")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="model/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    prepared = load_and_prepare(
        args.data_dir, target_name=args.target, test_size=args.val_split, random_state=args.seed)
    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    y_train = prepared["y_train"]
    y_val = prepared["y_val"]
    scaler = prepared["scaler"]
    imputer = prepared["imputer"]
    label_encoder_y = prepared["label_encoder_y"]
    feature_columns = prepared["feature_columns"]
    task = prepared["task"]
    output_dim = prepared["output_dim"]

    input_dim = X_train.shape[1]
    hidden_dims = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=(
        output_dim if task == "classification" else 1), task=task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_loader, val_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, args.batch_size)

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pt_name = f"financial_model_{timestamp}.pt"
    meta = {
        "created_at": timestamp,
        "task": task,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_dir": args.data_dir,
    }

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(
            model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_loop(
            model, device, val_loader, criterion, task)
        print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {train_loss:.6f}  ValLoss: {val_loss:.6f}  ValAcc: {val_acc if val_acc is None else f'{val_acc:.4f}'}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "imputer": imputer,
                "feature_columns": feature_columns,
                "task": task,
                "label_encoder_y": (label_encoder_y.classes_.tolist() if label_encoder_y is not None else None),
                "metadata": meta
            }
            pt_path = os.path.join(args.output_dir, pt_name)
            torch.save(best_ckpt, pt_path)
            print(f"Saved best .pt -> {pt_path}")

    # Save metadata json
    meta_path = os.path.join(args.output_dir, f"metadata_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Also create a pickle wrapper for direct Python inference (.predict)
    # We'll pickle the wrapper object that contains the model_state_dict and preprocessing objects.
    label_classes = best_ckpt.get("label_encoder_y", None)
    wrapper = InferenceWrapper(
        model_state_dict=best_ckpt["model_state"],
        scaler=best_ckpt["scaler"],
        imputer=best_ckpt["imputer"],
        feature_columns=best_ckpt["feature_columns"],
        task=best_ckpt["task"],
        label_classes=label_classes,
        hidden_dims=hidden_dims,
        input_dim=input_dim,
        output_dim=(output_dim if task == "classification" else 1)
    )
    pkl_path = os.path.join(args.output_dir, "trained_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(wrapper, f)
    print(f"Saved pickled inference wrapper -> {pkl_path}")

    print("Training complete. Artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
