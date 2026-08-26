import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=3,
            dilation=1,
            dropout=0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = x[:, :, :residual.size(2)]
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class ResidualTCNModel(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=128,
            dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            TCNBlock(hidden_size, dilation=1, dropout=dropout),
            TCNBlock(hidden_size, dilation=2, dropout=dropout),
            TCNBlock(hidden_size, dilation=4, dropout=dropout),
            TCNBlock(hidden_size, dilation=8, dropout=dropout),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.head(x)

        return x.squeeze(-1)
