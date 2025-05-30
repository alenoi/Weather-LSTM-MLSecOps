# src/utils/utils.py

import torch

def check_cuda():
    """
    Kiírja a CUDA elérhetőségét, verzióját és az aktuális device-ot.
    """
    print("CUDA elérhető:", torch.cuda.is_available())
    print("CUDA verzió:", torch.version.cuda)
    print("PyTorch build CUDA-támogatással:", torch.backends.cudnn.enabled)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Alapértelmezett eszköz:", device)
    return device

class EarlyStopping:
    """
    Early stopping a tanítás során. Ha adott számú epochon keresztül nem javul a loss, megállítja a tanulást.
    """
    def __init__(self, patience=10, verbose=True, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epoch óta nincs javulás.", end=' ')
            if self.counter >= self.patience:
                self.early_stop = True
