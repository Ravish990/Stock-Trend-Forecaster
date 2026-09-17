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

        # Price prediction
        self.price_fc = nn.Linear(hidden_size, 1)

        # Bullish / bearish prediction
        self.direction_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Last time step
        last_output = output[:, -1, :]

        price = self.price_fc(last_output)

        direction = self.direction_fc(last_output)

        return price, direction