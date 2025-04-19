import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout
    ):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        out, hn = self.gru(x)
        out = self.fc(out[:, -1, :])
        out = torch.relu(out)
        return out


class SurrogateCaModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=280):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
