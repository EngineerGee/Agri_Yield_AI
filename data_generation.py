# src/data_generation.py
import numpy as np
import pandas as pd
import os

def generate_synthetic_data(n_samples=500, seed=42, out_path="data/synthetic_crop_data.csv"):
    np.random.seed(seed)
    # Features: rainfall(mm), avg_temp(C), soil_ph, fertilizer_kg_per_ha, seed_density (kg/ha), sunlight_hours/day
    rainfall = np.random.normal(600, 150, n_samples).clip(100, 1500)   # mm/year
    avg_temp = np.random.normal(22, 4, n_samples).clip(5, 40)         # degrees C
    soil_ph = np.random.normal(6.5, 0.7, n_samples).clip(4.5, 8.5)
    fertilizer = np.random.normal(100, 50, n_samples).clip(0, 400)    # kg/ha
    seed_density = np.random.normal(75, 20, n_samples).clip(10, 200)  # kg/ha
    sunlight = np.random.normal(7, 1.5, n_samples).clip(3, 12)        # hrs/day

    # Simulated true underlying relationship (nonlinear) + noise
    # base yield influenced positively by rainfall up to an optimum, temperature optimal ~22, fertilizer and sunlight positive but diminishing returns
    rain_effect = 0.01 * rainfall - 0.00001 * (rainfall - 800)**2
    temp_effect = -0.05 * (avg_temp - 22)**2 + 6
    soil_effect = 0.8 * (1 - np.abs(soil_ph - 6.5) / 2)  # drops if ph far from 6.5
    fert_effect = 0.05 * fertilizer / (1 + 0.01 * fertilizer)  # diminishing returns
    seed_effect = 0.03 * seed_density / (1 + 0.02 * seed_density)
    sun_effect = 0.5 * sunlight

    noise = np.random.normal(0, 0.5, n_samples)

    # yield in tonnes per hectare (t/ha)
    yield_t_ha = 0.5 + 1.8 * rain_effect + 1.2 * temp_effect + 1.5 * soil_effect + 2.0 * fert_effect + 1.0 * seed_effect + 0.3 * sun_effect + noise
    # clip to realistic bounds
    yield_t_ha = np.clip(yield_t_ha, 0.2, 12)

    df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_C": avg_temp,
        "soil_ph": soil_ph,
        "fertilizer_kg_ha": fertilizer,
        "seed_density_kg_ha": seed_density,
        "sunlight_hrs": sunlight,
        "yield_t_ha": yield_t_ha
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to {out_path} with {n_samples} rows.")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
