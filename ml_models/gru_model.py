import torch
import torchvision
import torch.nn as nn
from mpmath.identification import transforms
import torchvision.transforms as transforms
from sympy import sequence
import torchvision.transforms
from ml_models.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ml_models.jh_transforms import ToTensor, ScaleAndShiftX, ScaleY



def gru():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Hyper Parameters
    num_classes = 1
    num_epochs = 2
    batch_size = 16
    lr = 0.001

    feature_size = 42
    sequence_length = 12
    hidden_size = 128
    num_layers = 2