import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self._features_size = feature_size
        self._hidden_size = hidden_size
        self._num_layers = 3
        self._gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self._num_layers
        )

        self._linear = nn.Linear(in_features=self._hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)
        h0.requires_grad = True
        _, hn = self._gru(x, h0)
        out = self._linear(hn[0]).flatten()

        return out


