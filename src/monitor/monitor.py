import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset

# 1. Konfigurációs útvonalak (.env-ben is lehet)
TRAIN_CSV = os.getenv("TRAIN_DATA_PATH", "data/processed/european_capitals_weather_combined.csv")
PRED_CSV = os.getenv("PRED_DATA_PATH", "data/processed/predictions.csv")

st.set_page_config(page_title="ML Model Monitor", layout="wide")
st.title("☁️ Weather LSTM Model Monitoring")

# 2. Dummy adatok létrehozása ha hiányzik valamelyik fájl
def generate_dummy_data(size=100):
    dates = pd.date_range(datetime.today() - timedelta(days=size), periods=size).to_pydatetime().tolist()
    return pd.DataFrame({
        "date": dates,
        "Budapest_tmax": np.random.uniform(10, 30, size),
        "Budapest_tmin": np.random.uniform(0, 15, size),
        "predicted_tmax": np.random.uniform(10, 30, size),
    })

# 3. Adatok betöltése vagy dummy generálása
def load_or_generate(path: str, is_prediction=False) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.warning(f"⚠️ Nem található a fájl: `{path}`, generált dummy adatokkal helyettesítve.")
        df = generate_dummy_data()
        if is_prediction:
            df["Budapest_tmax"] = df["predicted_tmax"] + np.random.normal(0, 1, len(df))
        return df

ref_data = load_or_generate(TRAIN_CSV)
prod_data = load_or_generate(PRED_CSV, is_prediction=True)

# 4. Adat preview
st.subheader("📊 Reference (training) data sample")
st.dataframe(ref_data.head(), use_container_width=True)

st.subheader("📈 Production (predictions) data sample")
st.dataframe(prod_data.head(), use_container_width=True)

# 5. Report generálása
report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
report.run(reference_data=ref_data, current_data=prod_data)

# 6. HTML megjelenítés
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
    report.save_html(tmp.name)
    html = open(tmp.name, "r").read()

st.subheader("✅ Monitoring Report")
st.components.v1.html(html, height=1000, scrolling=True)
