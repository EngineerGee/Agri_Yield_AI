# src/predict.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

def predict_single(sample_dict, model_path="models/model_latest.h5", scaler_path="models/scaler.joblib"):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    cols = ["rainfall_mm","avg_temp_C","soil_ph","fertilizer_kg_ha","seed_density_kg_ha","sunlight_hrs"]
    x = np.array([[sample_dict[c] for c in cols]])
    x_s = scaler.transform(x)
    pred = model.predict(x_s, verbose=0)[0,0]
    return float(pred)

if __name__ == "__main__":
    # example
    example = {
        "rainfall_mm": 700,
        "avg_temp_C": 21,
        "soil_ph": 6.4,
        "fertilizer_kg_ha": 120,
        "seed_density_kg_ha": 80,
        "sunlight_hrs": 7.5
    }
    pred = predict_single(example)
    print(f"Predicted yield (t/ha): {pred:.2f}")
