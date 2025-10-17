# src/utils.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path="data/synthetic_crop_data.csv", test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_csv(path)
    X = df.drop(columns=["yield_t_ha"])
    y = df["yield_t_ha"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # create validation from train_full
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=relative_val, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test, save_path="models/scaler.joblib"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Saved scaler to {save_path}")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
