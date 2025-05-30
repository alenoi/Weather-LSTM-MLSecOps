# src/predict.py

import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model.lstm import LSTMModel

# --- KONFIG (módosítsd igény szerint) ---
MODEL_PATH = "./models/lstm_latest.pt"  # vagy legutóbbi modelled neve
DATA_PATH = "./data/processed/new_input.csv"  # predikcióra használt input (időszak, stb.)
SCALER_PATH = None  # ha scalert mentettél volna
N_STEPS = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(input_size, hidden_size, num_layers, model_path=MODEL_PATH, device=DEVICE):
    model = LSTMModel(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def preprocess_features(df, scaler=None):
    # Figyelj, hogy ugyanazokat a feature-öket/scalert használd, mint tanításkor!
    features = df.drop(columns=["time"]).values  # vagy amit a train-ben csináltál
    if scaler is not None:
        features = scaler.transform(features)
    else:
        features = StandardScaler().fit_transform(features)
    return features

def create_windows(features, n_steps):
    X = []
    for i in range(n_steps, len(features)):
        X.append(features[i - n_steps:i])
    return np.array(X)

def predict():
    # --- 1. Adat betöltése ---
    df = pd.read_csv(DATA_PATH)
    features = preprocess_features(df)  # ugyanaz mint train!

    # --- 2. Window-k előállítása ---
    X = create_windows(features, N_STEPS)

    # --- 3. Modell betöltése: add meg a pontos méreteket vagy töltsd MLflow-ból a configot!
    input_size = X.shape[2]
    hidden_size = 64  # ahogy a train-ben használtad
    num_layers = 2

    model = load_model(input_size, hidden_size, num_layers)

    # --- 4. Előrejelzés ---
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    # --- 5. Mentés/kiírás ---
    output = pd.DataFrame({
        "time": df["time"].iloc[N_STEPS:].values,
        "prediction": preds
    })
    output.to_csv("./data/processed/predictions.csv", index=False)
    print("Predictions saved to ./data/processed/predictions.csv")

if __name__ == "__main__":
    predict()
