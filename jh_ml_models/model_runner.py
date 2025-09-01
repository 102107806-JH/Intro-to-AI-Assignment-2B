import numpy
import torch
from jh_ml_models.data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader
from jh_ml_models.gru_model import GRU
import numpy as np
from torch import nn
from jh_ml_models.test_and_train import train_model, test_model

def model_runner(model, loss_function, optimizer, batch_size, num_epochs, scats_site_number, device):
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=3, selected_scats_site=scats_site_number)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Untrained Test __________")
    test_model(test_loader, model, loss_function, device)
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}\n-------------")
        train_model(train_loader, model, loss_function, optimizer=optimizer, device=device)
        test_model(test_loader, model, loss_function, device=device)
        print()

