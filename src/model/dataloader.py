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

# --- DATASET ---
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- FŐ FUNKCIÓ: loader + (ha kell: adatgeneráló pipeline) ---
def create_dataloaders(
    n_steps=7,
    batch_size=64,
    target_column="Budapest_tmax",
    data_file="../data/processed/european_capitals_weather_combined.csv",
    scaler_path=None,  # ha akarod menteni/újrahasználni a scalert
    test_size=0.2,
    verbose=True,
):
    # Ellenőrizd a létezést, ha nincs: raise/hívj adatgenerátort stb.
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Nincs előkészített adat: {data_file}")

    df = pd.read_csv(data_file)
    if target_column not in df.columns:
        raise ValueError(f"A megadott célváltozó ({target_column}) nem található a DataFrame-ben!")

    raw_features = df.drop(columns=["time"])
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(raw_features), columns=raw_features.columns)
    cleaned_features = scaled_features.dropna(axis=1)
    features = cleaned_features.values
    target_series = df[target_column].values

    # Sliding window
    X, y = [], []
    for i in range(n_steps, len(features)):
        X.append(features[i - n_steps:i])
        y.append(target_series[i])

    X, y = np.array(X), np.array(y)

    if verbose:
        print(f"[Dataloader] X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    train_dataset = WeatherDataset(X_train, y_train)
    test_dataset = WeatherDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[2]
    return train_loader, test_loader, input_size

# --- Teszt futtatás ---
if __name__ == "__main__":
    train_loader, test_loader, input_size = create_dataloaders()
    print("Train/test loader & input_size ready!")
