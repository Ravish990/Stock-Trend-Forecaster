from torch import nn
from sklearn.preprocessing import StandardScaler
import numpy as np


class CrossSectionalScaler:
    """
    Standardizes stock features cross-sectionally per date to remove macro collinearity,
    and scales macro features across time.
    """
    def __init__(self, stock_features=None, macro_features=None):
        self.stock_features = stock_features or []
        self.macro_features = macro_features or []
        self.macro_scaler = StandardScaler()

    def fit(self, df, train_mask):
        train_macro = df.loc[train_mask, self.macro_features]
        self.macro_scaler.fit(train_macro)
        return self

    def transform_df(self, df):
        df = df.copy()
        grouped = df.groupby("Date")[self.stock_features]
        means = grouped.transform("mean")
        stds = grouped.transform("std")
        df[self.stock_features] = ((df[self.stock_features] - means) / (stds + 1e-6)).clip(-5.0, 5.0)

        scaled_macro = self.macro_scaler.transform(df[self.macro_features])
        df[self.macro_features] = np.clip(scaled_macro, -5.0, 5.0)
        return df

    def transform(self, array_2d):
        arr = array_2d.copy()
        return np.clip(arr, -5.0, 5.0)


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
        num_layers=2,
        dropout=0.5
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Regression head
        self.return_fc = nn.Linear(hidden_size, 1)

        # Deeper direction head — separate capacity for classification
        self.direction_fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)

        ret = self.return_fc(last_output)
        direction = self.direction_fc(last_output)

        return ret, direction


class TemporalAttention(nn.Module):
    """
    Temporal self-attention over the 20-day lookback window.
    Allows the network to focus on turning points, breakouts, or momentum shifts.
    """
    def __init__(self, hidden_size):
        super().__init__()
        import torch
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_size)
        import torch
        scores = self.attn(lstm_outputs)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)  # (batch_size, hidden_size)
        return context, weights


class AttentionLSTMModel(nn.Module):
    """
    Upgraded Multi-Task Forecasting Network:
    - 64-unit 2-layer LSTM with Temporal Attention
    - LayerNorm + 0.25 Dropout to preserve directional variance
    - Shared representation trunk + dual regression & classification heads
    """
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.25
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Shared trunk
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        # 1. Regression head (5-day log return)
        self.return_fc = nn.Linear(32, 1)

        # 2. Direction head (upward probability logit)
        self.direction_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        context, _ = self.attention(outputs)
        context = self.layer_norm(context)
        context = self.dropout(context)

        shared = self.shared_fc(context)
        ret = self.return_fc(shared)
        direction = self.direction_fc(shared)

        return ret, direction