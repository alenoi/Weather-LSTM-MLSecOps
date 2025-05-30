# src/constants.py

FEATURE_COLUMNS = [
    "tavg", "tmin", "tmax", "prcp", "wspd"
    # Ide írd be a többi feature nevét is, ha vannak extrák!
]

DEFAULT_MODEL_NAME = "Weather-LSTM-Model"
MLFLOW_EXPERIMENT = "Weather-LSTM"
N_STEPS = 7          # Sliding window méret – ha mindenhol ugyanaz
BATCH_SIZE = 64      # Default batch méret – ha mindenhol ugyanaz

# Ha pickle-ölsz skálázót vagy feature oszlopokat, akkor ezek is mehetnek ide:
SCALER_PATH = "./models/scaler.pkl"
FEATURE_PATH = "./models/feature_columns.pkl"

# (Opcionális, ha sok helyen kell)
TARGET_COLUMN = "Budapest_tmax"
