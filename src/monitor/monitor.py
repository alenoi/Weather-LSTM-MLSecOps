# src/monitor/monitor.py

import os
import tempfile

import streamlit as st
import pandas as pd

# 1. Evidently Report & Presets API
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset

# 2. Konfigurációs útvonalak (.env-ben is beállíthatod)
TRAIN_CSV = os.getenv("TRAIN_DATA_PATH", "data/processed/weather_cleaned.csv")
PRED_CSV = os.getenv("PRED_DATA_PATH", "data/processed/predictions.csv")

# 3. Oldal beállítása
st.set_page_config(page_title="ML Model Monitor", layout="wide")
st.title("☁️ Weather LSTM Model Monitoring")

# 4. Adatok betöltése
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

try:
    ref_data = load_data(TRAIN_CSV)
    prod_data = load_data(PRED_CSV)
except FileNotFoundError as e:
    st.error(f"Nem található az adatfájl: {e.filename}")
    st.stop()

# 5. Adat-előnézet
st.subheader("Reference (training) data sample")
st.dataframe(ref_data.head(), use_container_width=True)

st.subheader("Production (predictions) data sample")
st.dataframe(prod_data.head(), use_container_width=True)

# 6. Report összeállítása és futtatása
report = Report(metrics=[ DataDriftPreset(), RegressionPreset() ])
report.run(reference_data=ref_data, current_data=prod_data)

# 7. HTML export és beágyazás Streamlit-be
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
    report.save_html(tmp.name)
    html = open(tmp.name, "r").read()

st.subheader("✅ Monitoring Report")
st.components.v1.html(html, height=1000, scrolling=True)
