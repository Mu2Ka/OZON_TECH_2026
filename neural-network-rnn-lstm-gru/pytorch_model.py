import torch
from torch import nn


class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        output, hidden = self.rnn(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.linear(last_output).squeeze(-1)


class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.15, num_layers=2)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        last_output = output[:, -1, :]

        last_output = self.norm(last_output)
        return self.head(last_output).squeeze(-1)


class ImprovedGRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)

        output, hidden = self.gru(x)
        last_hidden = hidden[-1]

        prediction = self.output_layer(last_hidden)
        return prediction.squeeze(-1)


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.linear(last_output).squeeze(-1)
