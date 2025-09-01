from jh_ml_models.model_runner import model_runner
from torch.utils.data import DataLoader
from jh_ml_models.gru_model import GRU
import numpy as np
from torch import nn
from jh_ml_models.test_and_train import train_model, test_model
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    batch_size = 16
    lr = 5e-5
    hidden_size = 16
    num_epochs = 10

    model = GRU(feature_size=4, hidden_size=hidden_size, num_layers=3 ,device=device).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scats_site_number = 970

    model_runner(model, loss_function, optimizer, batch_size, num_epochs, scats_site_number, device=device)