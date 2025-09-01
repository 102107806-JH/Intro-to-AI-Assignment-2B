import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, device):
        super().__init__()
        self._features_size = feature_size
        self._hidden_size = hidden_size
        self._device = device
        self._num_layers = num_layers
        self._gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self._num_layers
        )

        self._linear = nn.Linear(in_features=self._hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size).to(self._device)
        gru_out, hn = self._gru(x, h0)

        # gru_out is in form (batch, sequence_index, features). The below takes
        # the final sequence features across all batches #
        out = self._linear(gru_out[:, -1, :]).flatten()

        return out


