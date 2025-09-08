from torch import nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, feature_size, sequence_length, kernel_size, c1_out_channels, device):
        conv1d_Lout = sequence_length - kernel_size + 1  # The length of the first output vector of the convolutional layer #
        conv2d_Lout = conv1d_Lout - kernel_size + 1  # The length of the first output vector of the convolutional layer #

        if conv2d_Lout < 1:  # The current sized kernel would result in an output of the second convolutional layer that has a length of under 1. This is impossible #
            raise Exception("Kernel size to large")
        super().__init__()
        self._feature_size = feature_size  # The number of features per time step #
        self._sequence_length = sequence_length  # The sequence length (number of time steps) that the network takes as input #
        self._kernel_size = kernel_size  # How many timesteps that the kernel covers #
        self._device = device  # CPU or GPU #
        # Init the conv layers
        self._conv1d_1 = nn.Conv1d(in_channels=feature_size,
                                   out_channels=c1_out_channels,
                                   kernel_size=kernel_size,
                                   stride=1)
        self._conv1d_2 = nn.Conv1d(in_channels=c1_out_channels,
                                   out_channels=1,
                                   kernel_size=kernel_size,
                                   stride=1)
        # Init the output linear layers
        self._linear = nn.Linear(in_features=conv2d_Lout,
                                 out_features=1)
        self._transform_dict = {} # Transformation dictionary that can be set using a setter at a later date. This is just a convenient way to store transformations for use during inference #


    def forward(self, x):
        if x.shape[1] != self._sequence_length: # Ensure the input has the correct number of time steps in the sequence #
            raise Exception("Expecting sequence length of " + str(self._sequence_length) + " but got " + str(x.shape[1]) + ". Ensure that data inputted to the model has the correct sequence length.")

        x = x.permute(0, 2, 1)  # Align the dimensions as per documentation #

        # Pass input through the network
        c1_out = F.tanh(self._conv1d_1(x))
        c2_out = F.tanh(self._conv1d_2(c1_out))
        out = self._linear(c2_out)
        out = out.flatten()  # Flatten the output #
        return out

    @property
    def transform_dict(self):  # Allows retrival of the transform dictionary #
        return self._transform_dict

    @transform_dict.setter
    def transform_dict(self, transform_dict):  # Allows setting of the transform dictionary #
        self._transform_dict = transform_dict
