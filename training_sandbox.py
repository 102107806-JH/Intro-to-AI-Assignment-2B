import torch
from torch import nn
from jh_ml_models.gru_model import GRU
from jh_ml_models.model_fitter import Model_Fitter

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    batch_size = 16
    lr = 5e-5
    hidden_size = 16
    num_epochs = 50

    model = GRU(feature_size=4, hidden_size=hidden_size, num_layers=6 ,device=device).to(device)
    train_loss_function = nn.MSELoss()
    test_loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scats_site_number = 970

    fitter = Model_Fitter(model=model,
                          train_loss_function=train_loss_function,
                          test_loss_function=test_loss_function,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          num_epochs=num_epochs,
                          device=device)

    fitter.fit_model(scats_site_number)

