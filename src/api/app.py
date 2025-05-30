import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

from constants import FEATURE_COLUMNS
from MLModel import WeatherModel

# Környezet
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")

# Modell betöltése Registry-ből
model = WeatherModel.load_from_registry(MODEL_NAME)

app = FastAPI()

class ForecastInput(BaseModel):
    data: list

class ForecastOutput(BaseModel):
    prediction: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ForecastOutput)
def predict(req: ForecastInput):
    df = pd.DataFrame([req.data], columns=FEATURE_COLUMNS)
    pred = model.predict(df)
    return ForecastOutput(prediction=pred[0])
