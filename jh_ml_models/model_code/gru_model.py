import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, feature_size, sequence_length, hidden_size, num_layers, device):
        super().__init__()
        self._feature_size = feature_size  # The number of features per time step #
        self._sequence_length = sequence_length  # The sequence length (number of time steps) that the network takes as input #
        self._hidden_size = hidden_size  # The size of the hidden state vector in the gru #
        self._device = device  # CPU or GPU #
        self._num_layers = num_layers  # Number of layers in the GRU #
        # Init the gru
        self._gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=self._num_layers
        ).to(self._device)
        self._linear = nn.Linear(in_features=self._hidden_size, out_features=1).to(self._device)  # Init the linear layer #
        self._transform_dict = {}  # Transformation dictionary that can be set using a setter at a later date. This is just a convenient way to store transformations for use during inference #


    def forward(self, x):
        if x.shape[1] != self._sequence_length:  # Ensure the input has the correct number of time steps in the sequence #
            raise Exception("Expecting sequence length of " + str(self._sequence_length) + " but got " + str(x.shape[1]) + ". Ensure that data inputted to the model has the correct sequence length.")

        batch_size = x.shape[0]
        h0 = torch.zeros(self._num_layers, batch_size, self._hidden_size).to(self._device)  # Setting the hidden state vector to zeros #
        gru_out, hn = self._gru(x, h0)  # Passing the input and the hidden state through the gru #

        # gru_out tensor is structured as follows (batch, sequence_index, features). Extracting the final sequence as the output #
        out = self._linear(gru_out[:, -1, :]).flatten()

        return out

    @property
    def transform_dict(self):  # Allows retrival of the transform dictionary #
        return self._transform_dict

    @transform_dict.setter
    def transform_dict(self, transform_dict):  # Allows setting of the transform dictionary #
        self._transform_dict = transform_dict