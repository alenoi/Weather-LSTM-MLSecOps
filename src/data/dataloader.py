# src/data/dataloader.py

import os
import time
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# --- DATASET CLASS ---
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- DATALOADER FUNKCIÓ ---
def create_dataloaders(
    n_steps=7,
    batch_size=64,
    target_column="Budapest_tmax",
    csv_path="data/processed/european_capitals_weather_combined.csv",
    scaler=None,
    return_scaler=False,
    test_size=0.2,
    shuffle=False,
    random_state=42,
):
    """
    - n_steps: sliding ablak hossza
    - batch_size: loader batch mérete
    - target_column: előrejelzendő target oszlop neve
    - csv_path: betöltendő feature CSV útvonala
    - scaler: meglévő StandardScaler példány, ha van (különben fit-el újat)
    - return_scaler: ha True, visszaadja a scalert és a feature columnt is
    """
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    if target_column not in df.columns:
        raise ValueError(f"Célváltozó '{target_column}' nem található a DataFrame-ben!")

    feature_columns = [col for col in df.columns if col != target_column]
    raw_features = df[feature_columns].values
    y = df[target_column].values

    # --- Scaling ---
    scaler = scaler or StandardScaler()
    X_scaled = scaler.fit_transform(raw_features)
    features = pd.DataFrame(X_scaled, columns=feature_columns).values

    # --- Sliding ablak (LSTM input) ---
    X_windowed = []
    y_windowed = []
    for i in range(n_steps, len(features)):
        X_windowed.append(features[i - n_steps:i])
        y_windowed.append(y[i])
    X_windowed = np.array(X_windowed)
    y_windowed = np.array(y_windowed)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_windowed, y_windowed, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    train_dataset = WeatherDataset(X_train, y_train)
    test_dataset = WeatherDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]

    if return_scaler:
        return train_loader, test_loader, input_size, scaler, feature_columns
    return train_loader, test_loader, input_size

# --- Tesztfutás ---
if __name__ == "__main__":
    loaders = create_dataloaders(return_scaler=True)
    train_loader, test_loader, input_size, scaler, feature_columns = loaders
    print("Train/test loader & input_size ready!")
    print("feature_columns:", feature_columns)
