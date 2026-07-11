# predictor.py
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("models", "trained_model.pkl")


class FinancialPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Please run train_model.py first.")
        self.model = joblib.load(MODEL_PATH)

    def predict_scheme(self, features: dict):
        """
        features = {
            "population_density": float,
            "avg_income": float,
            "agriculture_dependence": float,
            "female_population_ratio": float
        }
        """
        input_vector = np.array([
            features["population_density"],
            features["avg_income"],
            features["agriculture_dependence"],
            features["female_population_ratio"]
        ]).reshape(1, -1)

        prediction = self.model.predict(input_vector)[0]
        return prediction
