# 🌦️ Weather-LSTM-MLSecOps

Egy gépi tanuláson alapuló időjárás-előrejelző rendszer, amely LSTM neurális hálózatra épül, és modern MLOps eszközök segítségével valósítja meg az automatizált betanítást, verziókövetést és monitorozást. A cél: egy teljes pipeline kialakítása az adatfeldolgozástól a predikciók kiértékeléséig.

## 🔧 Fő funkciók

- LSTM-alapú regressziós modell hőmérséklet-előrejelzéshez
- Adatfeldolgozás `pandas` és `scikit-learn` segítségével
- Modell mentése és verziókezelése **MLflow**-val
- REST API szolgáltatás **FastAPI**-val
- **Streamlit** dashboard az előrejelzések és metrikák vizualizálására
- **EvidentlyAI** alapú adatsodródás- és teljesítménymonitorozás
- Automatizált újratanítás **Airflow DAG** használatával
- **Docker**-alapú, moduláris konténerizált környezet

## 🧠 Modellfelépítés

A modell egy többrétegű **LSTM (Long Short-Term Memory)** hálózat, amely múltbeli időjárási adatok alapján becsli meg a következő időlépés maximális hőmérsékletét.

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

A következő komponensek érhetők el:

- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
- **API (FastAPI)**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Streamlit dashboard**: [http://localhost:8501](http://localhost:8501)
- **Airflow UI**: [http://localhost:8081](http://localhost:8081)

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

A Streamlit alkalmazás vizuálisan megjeleníti:

- Az aktuális előrejelzést
- A predikció és a valós érték összehasonlítását
- Metrikákat (MAE, RMSE, R²)

## 🛰️ Airflow pipeline

A `retrain_pipeline.py` fájlban található DAG automatikusan:

1. Betölti az új adatokat  
2. Előfeldolgozást végez  
3. Újratanítja a modellt  
4. Logolja az eredményeket MLflow-ba  

A pipeline manuálisan indítható az Airflow UI-ból is.

## 📈 Drift detektálás

Az EvidentlyAI segítségével folyamatosan monitorozzuk az input adatok eloszlását, és detektáljuk az esetleges driftet a predikciós teljesítmény romlásának korai észleléséhez.

## 📜 Követelmények

Ha nem Dockerben futtatnád:

```bash
pip install -r requirements.txt
```

## 📝 Licenc

MIT License – szabadon használható, módosítható és terjeszthető.

---

## 👨‍💻 Fejlesztő

**Panyi Tamás**  
MSc Data Science – Óbudai Egyetem  
[GitHub](https://github.com/alenoi)

---

## 🎯 Fejlesztési irányok

- [ ] Hyperparameter tuning automatikusan (pl. Optuna integrációval)  
- [ ] Értesítés emailben/SMS-ben drift esetén  
- [ ] Streamlit sötét/világos mód váltás  
