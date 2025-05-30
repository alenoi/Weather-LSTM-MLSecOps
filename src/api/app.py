import os
import pickle
import pandas as pd
import numpy as np
import torch
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Saját importok
from src.constants import (
    FEATURE_COLUMNS,          # List[str]
    DEFAULT_MODEL_NAME,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_N_STEPS,
    DEFAULT_DEVICE,
    SCALER_PATH,
    FEATURE_PATH
)
from src.model.lstm import LSTMModel  

# ---- Konfiguráció ----
MODEL_NAME      = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", DEFAULT_HIDDEN_SIZE))
MODEL_NUM_LAYERS = int(os.getenv("NUM_LAYERS", DEFAULT_NUM_LAYERS))
MODEL_N_STEPS    = int(os.getenv("N_STEPS", DEFAULT_N_STEPS))
MODEL_DEVICE     = DEFAULT_DEVICE
SCALER_PATH      = os.getenv("SCALER_PATH", SCALER_PATH)
FEATURE_PATH     = os.getenv("FEATURE_PATH", FEATURE_PATH)

# ---- API & MLflow ----
app = FastAPI()
client = MlflowClient()

# ---- Segédfüggvények ----
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_latest_model():
    try:
        client.get_registered_model(MODEL_NAME)
    except RestException:
        print(f"Model '{MODEL_NAME}' is not registered yet.")
        return None
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.pytorch.load_model(model_uri, map_location=MODEL_DEVICE)
    return model

# ---- Betöltés induláskor ----
scaler = load_pickle(SCALER_PATH)
feature_columns = load_pickle(FEATURE_PATH)
model = load_latest_model()

# ---- Pydantic input/output ----
class ForecastInput(BaseModel):
    data: list  # Flattened N_STEPS * N_FEATURES (sliding window)

class ForecastOutput(BaseModel):
    prediction: float

# ---- Endpoints ----

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ForecastOutput)
def predict(input_data: ForecastInput):
    if model is None or scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model, scaler vagy feature list hiányzik.")

    if len(input_data.data) != MODEL_N_STEPS * len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Input length must be n_steps * n_features = {MODEL_N_STEPS * len(feature_columns)}"
        )
    # 1. Formázás sliding window-ra
    X_window = np.array(input_data.data).reshape(1, MODEL_N_STEPS, len(feature_columns))
    # 2. Scaling feature-wise (az összes idősíkra ugyanazzal a scalerrel!)
    X_scaled = scaler.transform(X_window.reshape(-1, len(feature_columns))).reshape(1, MODEL_N_STEPS, len(feature_columns))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(MODEL_DEVICE)

    # 3. Modell predikció
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        pred_value = float(y_pred.cpu().numpy().flatten()[0])

    return ForecastOutput(prediction=pred_value)
