import torch
from torch import nn
from jh_ml_models.model_code.gru_model import GRU
from jh_ml_models.model_code.tcn_model import TCN
from jh_ml_models.model_fitting.model_fitter import Model_Fitter

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters (Common)
    batch_size = 5
    lr = 5e-5
    num_epochs = 3
    sequence_length = 12

    # Hyper parameters (GRU)
    hidden_size = 128
    num_layers = 3

    # Hyper parameters (TCN)
    kernel_size = 4
    c1_out_channels = 6

    model = GRU(feature_size=4, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, device=device).to(device)
    #model = TCN(feature_size=4, sequence_length=sequence_length, kernel_size=kernel_size, c1_out_channels=c1_out_channels, device=device).to(device)
    train_loss_function = nn.MSELoss()
    test_loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scats_site_number = 'ALL'

    fitter = Model_Fitter(model=model,
                          train_loss_function=train_loss_function,
                          test_loss_function=test_loss_function,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          num_epochs=num_epochs,
                          sequence_length=sequence_length,
                          device=device,
                          model_save_path="saved_models/gru.pth")

    fitter.fit_model(scats_site_number)

