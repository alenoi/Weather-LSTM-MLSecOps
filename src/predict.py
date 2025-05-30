# src/predict.py

import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.model.lstm import LSTMModel

# --- KONFIG: Ezeket a tanításkor elmentett értékekkel kell egyeztetni ---
MODEL_PATH = os.getenv("MODEL_PATH", "./models/best_lstm_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")
FEATURE_PATH = os.getenv("FEATURE_PATH", "./models/feature_columns.pkl")
N_STEPS = int(os.getenv("N_STEPS", 7))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 64))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", 2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_columns(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model(model_path, input_size, hidden_size, num_layers, device):
    model = LSTMModel(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_single(input_sequence, model, scaler, feature_columns, device=DEVICE):
    """
    input_sequence: np.ndarray shape (n_steps, n_features)
    Returns: float (predicted value)
    """
    X_scaled = scaler.transform(input_sequence)
    X_tensor = torch.tensor(X_scaled.reshape(1, N_STEPS, len(feature_columns)), dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_tensor)
    return float(y_pred.cpu().numpy().flatten()[0])

def predict_from_csv(input_csv, output_csv=None):
    # --- Load dependencies ---
    scaler = load_scaler(SCALER_PATH)
    feature_columns = load_feature_columns(FEATURE_PATH)
    df = pd.read_csv(input_csv)

    # Ellenőrizzük, hogy van elég sor a sliding window-hoz
    if len(df) < N_STEPS:
        raise ValueError(f"Legalább {N_STEPS} sor kell a predikcióhoz!")

    # Minden predikcióhoz az utolsó N_STEPS sort használjuk fel (moving window)
    predictions = []
    model = None  # csak egyszer töltjük be!
    for i in range(N_STEPS, len(df) + 1):
        window = df.iloc[i - N_STEPS:i][feature_columns].values
        if model is None:
            model = load_model(MODEL_PATH, window.shape[1], HIDDEN_SIZE, NUM_LAYERS, DEVICE)
        pred = predict_single(window, model, scaler, feature_columns, device=DEVICE)
        predictions.append(pred)
    result_df = df.iloc[N_STEPS - 1:].copy()
    result_df["prediction"] = predictions
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Predikciók mentve: {output_csv}")
    return result_df

# --- Példa parancssori futtatás ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSTM időjárás predikció batch módban.")
    parser.add_argument("--input", required=True, help="Bemeneti CSV a feature-ökkel")
    parser.add_argument("--output", required=False, help="Kimeneti CSV a predikciókkal")

    args = parser.parse_args()
    df_pred = predict_from_csv(args.input, args.output)
    print(df_pred.head())
