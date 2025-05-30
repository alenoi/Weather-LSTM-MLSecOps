# src/train.py

import os
import time
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pytorch

from src.model.lstm import LSTMModel
from src.data.dataloader import create_dataloaders
from src.utils.early_stopping import EarlyStopping
import sys
sys.path.append('/app')


# --- Alapértelmezett konfiguráció ---
DEFAULT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "early_stopping_patience": 10,
    "num_epochs": 100,
    "model_dir": "./models",
    "plots_dir": "./plots",
    "mlflow_experiment": "Weather-LSTM",
    "registered_model_name": "Weather-LSTM-Model",
    "n_steps": 7,
    "target_column": "Budapest_tmax"
}

def train(config=None):
    """Futtatja a teljes tanítási pipeline-t. Paraméterezhető."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # 1. Adatbetöltés
    train_loader, test_loader, input_size, scaler, feature_columns = create_dataloaders(
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        target_column=config["target_column"],
        return_scaler=True
    )

    # 2. Modell példányosítás
    model = LSTMModel(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"]
    ).to(config["device"])

    # 3. Loss, optimizer, early stopping
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
    train_loss_history = []

    # 4. MLflow setup
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run(run_name="LSTM_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")):
        mlflow.log_params({
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "early_stopping_patience": config["early_stopping_patience"],
            "n_steps": config["n_steps"]
        })
        best_loss = float('inf')
        start_time = time.time()
        for epoch in range(config["num_epochs"]):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)
            train_loss_history.append(train_loss)
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Train loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print("Early stopping activated.")
                break

        elapsed_time = time.time() - start_time

        # 5. Kiértékelés
        test_mse, test_mae = evaluate(model, test_loader, config)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_mae", test_mae)

        # 6. Modell/Scaler/Feature Columns mentése (pickle)
        os.makedirs(config["model_dir"], exist_ok=True)
        model_path = os.path.join(config["model_dir"], f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

        # Pickle scaler és feature columns (API/predict miatt!)
        scaler_path = os.path.join(config["model_dir"], "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        feat_path = os.path.join(config["model_dir"], "feature_columns.pkl")
        with open(feat_path, "wb") as f:
            pickle.dump(feature_columns, f)

        print(f"Scaler saved at {scaler_path}")
        print(f"Feature columns saved at {feat_path}")

        # 7. Modell logolás/regisztrálás MLflow-ban
        try:
            mlflow.pytorch.log_model(
                model, artifact_path="model", registered_model_name=config["registered_model_name"]
            )
            print("Model registered in MLflow!")
        except Exception as e:
            print(f"Model log/registration failed: {e}")

        # 8. Loss ábra mentése
        save_training_curve(train_loss_history, config)
        print("Done. Training duration: {:.2f} seconds".format(elapsed_time))

def evaluate(model, test_loader, config):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config["device"])
            outputs = model(inputs).cpu().numpy()
            predictions.extend(outputs.flatten())
            actuals.extend(targets.numpy().flatten())
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f"Test MSE: {mse:.3f}, Test MAE: {mae:.3f}")
    return mse, mae

def save_training_curve(train_loss_history, config):
    os.makedirs(config["plots_dir"], exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_history, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(config["plots_dir"], f"train_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(fname)
    print(f"Loss plot saved: {fname}")

if __name__ == "__main__":
    train()
