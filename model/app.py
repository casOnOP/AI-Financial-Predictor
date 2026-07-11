from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb


def create_app(models_dir="models", data_csv="data/karnataka_dataset.csv"):
    app = Flask(__name__)
    CORS(app)

    MODEL_PATH = os.path.join(models_dir, "financial_model_gpu.xgb")
    ENCODER_PATH = os.path.join(models_dir, "label_encoder.pkl")
    FEATURES_PATH = os.path.join(models_dir, "feature_columns.json")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURES_PATH)):
        raise FileNotFoundError(
            "Model not found! Train first:\n"
            "python train_model_gpu.py --data data/karnataka_dataset.csv"
        )

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        FEATURE_COLS = json.load(f)

    target_encoder = encoders["target"]
    target_classes = list(target_encoder.classes_)

    base_df = pd.read_csv(data_csv)
    # Normalize district column
    for col in ["district", "District", "DISTRICT", "NAME_2", "name"]:
        if col in base_df.columns:
            base_df["district"] = base_df[col].astype(str).str.strip()
            break
    else:
        raise ValueError("No valid district column found in dataset!")

    # Normalize main crop
    if "main_crop" in base_df.columns:
        base_df["main_crop"] = base_df["main_crop"].astype(str).str.lower()
    else:
        raise ValueError("Missing 'main_crop' column in dataset!")

    def prepare_input(payload):
        echo = {}

        if "district" in payload:
            name = payload["district"].strip()
            row = base_df[base_df["district"].str.lower() == name.lower()]
            if row.empty:
                raise ValueError(f"Unknown district: {name}")
            row = row.iloc[0].to_dict()
            echo["district"] = name
        else:
            raise ValueError("Request must include 'district'")

        processed = dict(row)

        # Encode main_crop
        mc = processed["main_crop"]
        le_crop = encoders["main_crop"]
        try:
            processed["main_crop"] = int(le_crop.transform([mc])[0])
        except:
            processed["main_crop"] = 0

        X = np.array([[processed[c] for c in FEATURE_COLS]], dtype=np.float32)
        return echo, X, processed

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json(force=True)
            print("📌 Incoming payload:", payload)
            echo, X, row = prepare_input(payload)

            probs = booster.predict(xgb.DMatrix(X))[0]
            best_idx = int(np.argmax(probs))
            scheme = target_classes[best_idx]
            confidence = float(probs[best_idx])

            preview_keys = [
                "literacy_rate",
                "agri_dependence",
                "avg_income",
                "employment_rate",
                "bank_branches"
            ]

            preview = {}
            for k in preview_keys:
                val = row.get(k)
                if val is not None:
                    preview[k] = float(
                        val) if k != "bank_branches" else int(val)

            return jsonify({
                "district": echo["district"],
                "scheme": scheme,
                "confidence": round(confidence, 4),
                "data": preview
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    app = create_app()
    print("✅ GPU model loaded successfully!")
    app.run(host=args.host, port=args.port, debug=True)
