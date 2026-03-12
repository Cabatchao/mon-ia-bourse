"""Famille de modèles deep learning pour séries temporelles."""
from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = y[..., : x.shape[-1]]
        return x + y


class TCNModel(nn.Module):
    def __init__(self, n_features: int, channels: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.tcn = nn.Sequential(TCNBlock(channels, dilation=1), TCNBlock(channels, dilation=2), TCNBlock(channels, dilation=4))
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.tcn(self.in_proj(x))
        return self.head(y[:, :, -1]).squeeze(-1)


class CNNTimeSeriesModel(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.head(x).squeeze(-1)


class TransformerTSModel(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(self.proj(x))
        return self.head(y[:, -1, :]).squeeze(-1)


class HybridLSTMTransformerMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        y = self.transformer(y)
        return self.head(y[:, -1, :]).squeeze(-1)
