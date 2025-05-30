# 🌦️ Weather-LSTM-MLSecOps

Egy gépi tanuláson alapuló időjárás-előrejelző rendszer LSTM neurális hálózatokkal, amelyet modern MLOps eszközökkel támogatunk és teszünk újrataníthatóvá. A cél az időjárási adatok alapján történő hőmérséklet-előrejelzés, illetve az ehhez szükséges adatelőkészítés, tanítás, validálás és monitorozás teljes pipeline-jának kiépítése.

## 🚀 Funkciók

- LSTM-alapú regressziós modell hőmérséklet-előrejelzésre
- Adatfeldolgozás `pandas` és `scikit-learn` segítségével
- Modell mentés és verziókezelés **MLflow**-val
- REST API szolgáltatás FastAPI segítségével
- Streamlit dashboard az előrejelzések és metrikák megjelenítésére
- EvidentlyAI alapú drift detektálás
- Automatikus újratanítás **Airflow** DAG segítségével
- Docker alapú környezet minden komponenshez

## 🧠 Modell

A modell egy egyszerű, 3 rétegű **LSTM** hálózat, amely egy 1 dimenziós időjárási idősor alapján becsli meg a következő időlépés hőmérsékletét.

## 📁 Projekt struktúra

```
.
├── docker/
│   ├── api/
│   ├── airflow/
│   ├── mlflow/
│   └── streamlit/
├── notebooks/
├── scripts/
│   ├── train.py
│   ├── preprocess.py
│   └── predict.py
├── dags/
│   └── retrain_pipeline.py
├── app/
│   ├── main.py  # FastAPI app
│   └── utils.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🐳 Telepítés és futtatás

### 1. Klónozd a repót

```bash
git clone https://github.com/alenoi/Weather-LSTM-MLSecOps.git
cd Weather-LSTM-MLSecOps
```

### 2. Indítsd el a szolgáltatásokat

```bash
docker-compose up --build
```

A következő komponensek fognak elindulni:
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
- **API (FastAPI)**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Streamlit dashboard**: [http://localhost:8501](http://localhost:8501)
- **Airflow UI**: [http://localhost:8081](http://localhost:8081)
- **Evidently monitor**: [http://localhost:8502](http://localhost:8502) *(ha beállítottad)*

## 🔬 API példa

### `/predict` (POST)

```json
{
  "sequence": [12.3, 13.5, 15.1, 14.7, 13.2]
}
```

Válasz:
```json
{
  "predicted_temperature": 14.98
}
```

## 📊 Dashboard

A Streamlit alkalmazás vizuálisan mutatja be:
- Az aktuális előrejelzést
- A predikció és a valós érték összehasonlítását
- Metrikákat (MAE, RMSE, R²)

## 🛰️ Airflow pipeline

A `retrain_pipeline.py` fájlban található DAG automatikusan:
1. Betölti az új adatokat
2. Előfeldolgozza azokat
3. Újratanítja a modellt
4. Logolja az eredményeket MLflow-ba

A pipeline manuálisan is indítható az Airflow UI-ból.

## 📈 Drift detektálás

Az EvidentlyAI segítségével folyamatosan monitorozzuk az input adatok eloszlását és figyeljük az esetleges driftet.

## 📜 Követelmények

Ha nem Dockerrel futtatod:

```bash
pip install -r requirements.txt
```

## 📝 Licenc

MIT License. Használd, forgasd, forkold bátran!

---

## 👨‍💻 Fejlesztő

**Tomi Panyi**  
MSc Data Science – Óbudai Egyetem  
[GitHub](https://github.com/alenoi)

---

## 🎯 TODO / ötletek

- [ ] Hyperparameter tuning automatikusan (Optuna?)
- [ ] Email/SMS alert drift esetén
- [ ] Streamlit dark/light mód váltás
