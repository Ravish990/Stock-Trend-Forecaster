from torch import nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.4
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
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]
        last_output = self.layer_norm(last_output)
        last_output = self.dropout(last_output)

        ret = self.return_fc(last_output)
        direction = self.direction_fc(last_output)

        return ret, direction