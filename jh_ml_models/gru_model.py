import numpy
import torch
from jh_ml_models.data_loader import TrafficFlowDataSet
import numpy as np
def gru():
    numpy.set_printoptions(linewidth=400, precision=4)
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=4)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader

    for i in range(2):
        x, y = train_dataset[i]
        print(x)