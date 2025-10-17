# src/train_model.py
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from utils import load_data, scale_data

def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse",
                  metrics=["mae"])
    return model

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)

    model = build_model(X_train_s.shape[1])
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/model_latest.h5"

    history = model.fit(X_train_s, y_train,
                        validation_data=(X_val_s, y_val),
                        epochs=80,
                        batch_size=32,
                        verbose=2)

    model.save(checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

    # Plot training history
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training History")
    os.makedirs("screenshots", exist_ok=True)
    plt.savefig("screenshots/training_plot.png", bbox_inches="tight", dpi=150)
    print("Saved training plot to screenshots/training_plot.png")

    # quick evaluation
    test_metrics = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"Test metrics: loss (mse) = {test_metrics[0]:.4f}, mae = {test_metrics[1]:.4f}")

if __name__ == "__main__":
    main()
