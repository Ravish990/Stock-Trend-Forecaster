from torch import nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # nn.LSTM's internal dropout only applies BETWEEN stacked
        # layers, not to the final hidden state that feeds the output
        # heads. This dropout covers that gap.
        self.dropout = nn.Dropout(dropout)

        # Return prediction (renamed from price_fc: this head predicts
        # next_return, not raw next_close)
        self.return_fc = nn.Linear(hidden_size, 1)

        # Bullish / bearish prediction (raw logits, used with
        # BCEWithLogitsLoss — no Sigmoid here)
        self.direction_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Last time step
        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)
        ret = self.return_fc(last_output)

        direction = self.direction_fc(last_output)

        return ret, direction