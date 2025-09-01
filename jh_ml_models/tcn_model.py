import torch
from torch import nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, feature_size, sequence_length, kernel_size, device):
        conv1d_Lout = sequence_length - kernel_size + 1
        conv2d_Lout = conv1d_Lout - kernel_size + 1

        if conv2d_Lout < 1:
            raise("Kernel size to large")
        super().__init__()
        self._feature_size = feature_size
        self._kernel_size = kernel_size
        self._device = device

        self._conv1d_1 = nn.Conv1d(in_channels=feature_size,
                                   out_channels=feature_size,
                                   kernel_size=kernel_size,
                                   stride=1)

        self._conv1d_2 = nn.Conv1d(in_channels=feature_size,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   stride=1)

        self._linear = nn.Linear(in_features=conv2d_Lout,
                                 out_features=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Align the dimensions as per documenation #
        c1_out = F.tanh(self._conv1d_1(x))
        c2_out = F.tanh(self._conv1d_2(c1_out))
        out = self._linear(c2_out)
        out = out.flatten()
        return out
