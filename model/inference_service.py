# model/inference_service.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import warnings
import os
import json
import traceback as _traceback
from sklearn.preprocessing import LabelEncoder  # noqa: F401

# -----------------------
# App & Request/Response
# -----------------------
app = FastAPI(title="AI-Financial-Predictor Inference")

# Request accepts rows and optional district name (for returning metadata)


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    district: Optional[str] = None


# Per-row prediction returned
class RowPrediction(BaseModel):
    prediction: float
    recommended_scheme: Optional[str] = None
    confidence: Optional[float] = None  # percentage 0-100
    metrics: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    results: List[RowPrediction]


# -----------------------
# MLP definition (must match training)
# -----------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), output_dim=1, dropout=0.0):
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


# -----------------------
# Checkpoint loader + wrapper
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
PT_GLOB = str(BASE_DIR / "models" / "*.pt")
PKL_PATH = BASE_DIR / "models" / "trained_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
METADATA_GLOB = str(BASE_DIR / "models" / "metadata_*.json")


def find_latest_pt():
    pts = sorted(glob.glob(PT_GLOB), key=lambda p: Path(p).stat().st_mtime)
    return Path(pts[-1]) if pts else None


class WrapperFromCheckpoint:
    """Wraps a loaded checkpoint to provide predict(rows) -> original-scale preds"""

    def __init__(self, ck):
        # retrieve common entries, allow for multiple naming variants
        self.model_state = ck.get("model_state") or ck.get(
            "model_state_dict") or ck.get("state_dict") or {}
        self.scaler = ck.get("scaler", None)
        self.imputer = ck.get("imputer", None)
        self.feature_columns = ck.get("feature_columns", None)
        self.target_transform_meta = ck.get("target_transform_meta", ck.get(
            "metadata", {}).get("target_transform", {"type": "none"}))
        self.hidden_dims = tuple(
            ck.get("metadata", {}).get("hidden_dims", (256, 128)))
        # infer input_dim
        self.input_dim = ck.get("input_dim", None) or (
            len(self.feature_columns) if self.feature_columns is not None else None)
        self.output_dim = ck.get("output_dim", 1)

        if self.input_dim is None:
            raise RuntimeError(
                "Checkpoint missing input_dim and feature_columns; cannot build model."
            )

        # build and load model
        self._model = MLP(input_dim=self.input_dim,
                          hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        try:
            self._model.load_state_dict(self.model_state)
        except Exception as e:
            warnings.warn(f"Could not load model state_dict: {e}")
        self._model.eval()

    def _inverse_target(self, arr):
        if arr is None:
            return arr
        tt = self.target_transform_meta or {}
        ttype = tt.get("type", "none")
        if ttype == "none":
            return arr
        if ttype == "log":
            return np.expm1(arr)
        if ttype == "standardize":
            mu = tt.get("mean", 0.0)
            sigma = tt.get("scale", 1.0)
            return arr * sigma + mu
        if ttype == "robust":
            med = tt.get("median", 0.0)
            iqr = tt.get("iqr", 1.0)
            return arr * iqr + med
        if ttype == "winsorize":
            return arr
        return arr

    def predict_from_array(self, X_arr):
        # Force numeric conversion (already cleaned earlier)
        X = np.asarray(X_arr, dtype=float)

        if self.imputer is not None:
            X = self.imputer.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X).astype("float32")

        xb = torch.from_numpy(X)
        with torch.no_grad():
            out = self._model(xb).cpu().numpy().reshape(-1)

        return self._inverse_target(out)

    def predict(self, rows):
        """
        Accepts numpy array or a list of dicts.
        Ensures non-numeric fields (like district names) get coerced to NaN 
        and properly imputed.
        """
        # If numpy array directly
        if isinstance(rows, np.ndarray):
            return self.predict_from_array(rows)

        # If rows is a list of dicts from API
        if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
            df = pd.DataFrame(rows)

            # ensure missing columns exist
            for c in self.feature_columns:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[self.feature_columns]

            # ⭐ COERCE text fields to NaN so numeric pipeline works
            df = df.apply(pd.to_numeric, errors="coerce")

            return self.predict_from_array(df.values)

        # Generic DataFrame-like case
        try:
            df = pd.DataFrame(rows)

            for c in self.feature_columns:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[self.feature_columns]

            df = df.apply(pd.to_numeric, errors="coerce")

            return self.predict_from_array(df.values)

        except Exception as e:
            raise RuntimeError(
                "Unsupported input format to predict: " + str(e))


# -----------------------
# Model + label encoder + metadata load logic
# -----------------------
wrapper = None
pt_path = find_latest_pt()
if pt_path:
    print("Loading checkpoint from .pt ->", pt_path)
    ck = torch.load(str(pt_path), map_location="cpu")
    try:
        wrapper = WrapperFromCheckpoint(ck)
        print("Model wrapper created from .pt checkpoint.")
    except Exception as e:
        raise RuntimeError(f"Failed to build wrapper from checkpoint: {e}")
else:
    # fallback to older pickle
    if PKL_PATH.exists():
        print("No .pt found — loading pickled wrapper:", PKL_PATH)
        with open(PKL_PATH, "rb") as f:
            wrapper = pickle.load(f)
    else:
        raise RuntimeError(
            f"No model checkpoint (.pt) found at {BASE_DIR/'models'} and trained_model.pkl missing. Run training first.")

# load label encoder if available
label_encoder = None
if Path(LABEL_ENCODER_PATH).exists():
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        # If it's a sklearn LabelEncoder, good. If it's dict or other, keep as-is.
        print("Loaded label_encoder from", LABEL_ENCODER_PATH)
    except Exception as e:
        warnings.warn(f"Could not load label_encoder.pkl: {e}")
else:
    print("label_encoder.pkl not found; predictions will be numeric unless frontend supplies mapping.")

# load metadata files (metadata_*.json)
metadata_store = {}
for path in sorted(glob.glob(METADATA_GLOB)):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
            # Expecting a mapping of district -> metrics OR list of dicts
            # Normalize: if meta is dict and has 'district' keys, merge into metadata_store
            if isinstance(meta, dict):
                # if structure is { "districts": [ ... ] } or direct mapping by name
                if "districts" in meta and isinstance(meta["districts"], list):
                    for item in meta["districts"]:
                        name = item.get("district") or item.get(
                            "name") or item.get("district_name")
                        if name:
                            metadata_store[name.strip().lower()] = item
                else:
                    # try to use as mapping directly (district_name -> metrics)
                    for k, v in meta.items():
                        metadata_store[str(k).strip().lower()] = v
            elif isinstance(meta, list):
                for item in meta:
                    name = item.get("district") or item.get(
                        "name") or item.get("district_name")
                    if name:
                        metadata_store[name.strip().lower()] = item
    except Exception as e:
        warnings.warn(f"Failed reading metadata file {path}: {e}")

print(f"Loaded metadata entries: {len(metadata_store)}")

# Basic sanity check
feature_columns = getattr(wrapper, "feature_columns", None)
if feature_columns is None:
    raise RuntimeError(
        "Loaded model wrapper does not expose feature_columns. Cannot proceed.")


# -----------------------
# Utility: map numeric -> scheme using label encoder or rounding
# -----------------------
def numeric_to_scheme_and_confidence(pred_val):
    """
    If label_encoder is available, we will round pred_val to nearest integer,
    clip to the encoder class range, and inverse_transform to get scheme label.
    Confidence heuristic: 1 - abs(pred - rounded) (clamped 0..1), converted to percent.
    """
    scheme = None
    conf_pct = None

    if label_encoder is not None:
        try:
            rounded = int(np.rint(pred_val))
            classes = getattr(label_encoder, "classes_", None)
            if classes is not None:
                # classes_ length
                n = len(classes)
                # clip rounded to valid index range 0..n-1
                if rounded < 0:
                    rounded = 0
                if rounded >= n:
                    rounded = n - 1
                # Map index -> class label. If label_encoder used sklearn LabelEncoder,
                # classes_ are ordered such that encoded ints correspond to indices only if encoder used integer mapping.
                # We attempt inverse by index if classes_ is list-like and its entries are labels.
                # If label_encoder was trained with values mapping to [0..n-1], this should work.
                # Safer approach: if encoder was fit on numeric label IDs, you may need to adapt.
                scheme = str(classes[rounded])
            else:
                # If label_encoder is some other mapping dict
                try:
                    scheme = label_encoder.inverse_transform([rounded])[0]
                except Exception:
                    scheme = str(rounded)
            # confidence heuristic
            conf = 1.0 - min(1.0, abs(pred_val - np.rint(pred_val)))
            conf_pct = float(round(conf * 100.0, 2))
        except Exception:
            scheme = None
            conf_pct = None
    else:
        # no encoder: still provide rounded integer as "scheme id"
        rounded = int(np.rint(pred_val))
        scheme = f"scheme_id_{rounded}"
        conf = 1.0 - min(1.0, abs(pred_val - np.rint(pred_val)))
        conf_pct = float(round(conf * 100.0, 2))

    return scheme, conf_pct


# -----------------------
# API endpoints
# -----------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        rows = req.rows
        district_name = req.district
        if not isinstance(rows, list) or len(rows) == 0:
            raise HTTPException(
                status_code=400, detail='Request body must contain "rows": non-empty array'
            )

        preds = wrapper.predict(rows)
        preds = np.asarray(preds).reshape(-1)

        results = []
        for i, p in enumerate(preds):
            scheme, conf = numeric_to_scheme_and_confidence(float(p))

            # metrics lookup: use provided district_name OR try find district in row if present
            metrics = None
            # priority: explicit district param, then 'district' key inside row
            lookup_names = []
            if district_name:
                lookup_names.append(district_name)
            try:
                row = rows[i]
                if isinstance(row, dict):
                    # common fields
                    rn = row.get("district") or row.get(
                        "district_name") or row.get("districtName")
                    if rn:
                        lookup_names.append(rn)
            except Exception:
                pass

            found = None
            for name in lookup_names:
                if not name:
                    continue
                key = str(name).strip().lower()
                if key in metadata_store:
                    found = metadata_store[key]
                    break

            if found:
                metrics = found
            else:
                metrics = {}

            results.append(
                {
                    "prediction": float(p),
                    "recommended_scheme": scheme,
                    "confidence": conf,
                    "metrics": metrics,
                }
            )

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        tb = _traceback.format_exc()
        print("=== PREDICT ERROR ===")
        print(tb)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e), "trace": tb},
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_features": len(feature_columns)}
