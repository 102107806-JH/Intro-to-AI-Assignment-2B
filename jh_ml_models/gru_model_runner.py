import numpy
import torch
from jh_ml_models.data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader
from jh_ml_models.gru_model import GRU
import numpy as np
from torch import nn
from jh_ml_models.test_and_train import train_model, test_model

def gru_model_runner():
    numpy.set_printoptions(linewidth=400, precision=4)
    # Hyper parameters
    batch_size = 5
    lr = 5e-5
    hidden_size = 16


    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=12, selected_scats_site=970)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    x,y = dataset[0]
    print(x.shape)

    model = GRU(feature_size=4, hidden_size=hidden_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Untrained Test __________")
    test_model(test_loader, model, loss_function)
    print()

    for epoch in range(5):
        print(f"Epoch {epoch}\n-------------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

