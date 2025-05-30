import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Egyszerű LSTM regressziós modell.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]          # csak az utolsó időlépés (batch, hidden_size)
        out = self.fc(out)           # (batch, 1)
        return out

