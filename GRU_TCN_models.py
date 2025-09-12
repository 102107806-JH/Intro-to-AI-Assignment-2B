import torch
from torch import nn
from jh_ml_models.model_code.gru_model import GRU
from jh_ml_models.model_code.tcn_model import TCN
from jh_ml_models.model_fitting.model_fitter import Model_Fitter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = "tcn"

    # Hyperparameters (Common)
    num_epochs = 10
    sequence_length = 12

    # Hyperparameters (GRU)
    #lr = 0.00001
    #batch_size = 128
    #hidden_size = 96
    #num_layers = 5

    # Hyperparameters (TCN)
    lr = 0.001
    batch_size = 48
    kernel_size = 3
    c1_out_channels = 1
    if mode == "gru":
        model = GRU(feature_size=4, sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, device=device).to(device)
    elif mode == "tcn":
        model = TCN(feature_size=4, sequence_length=sequence_length, kernel_size=kernel_size, c1_out_channels=c1_out_channels, device=device).to(device)
    else:
        raise Exception("Invalid mode")
    train_loss_function = nn.MSELoss() #nn.L1Loss()
    test_loss_function = nn.MSELoss() #nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scats_site_number = 'ALL'

    split_proportions = {
        "train": 0.8,
        "test": 0.1,
        "validation": 0.1
    }
    split_proportions["discard"] = 0#1 - split_proportions["train"] - split_proportions["test"] - split_proportions["validation"]

    fitter = Model_Fitter(model=model,
                          train_loss_function=train_loss_function,
                          test_loss_function=test_loss_function,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          num_epochs=num_epochs,
                          sequence_length=sequence_length,
                          device=device,
                          split_proportions=split_proportions,
                          validate=True,
                          save_directory="saved_models",
                          save_name=mode)

    metric_dictionary = fitter.fit_model(scats_site_number)

    print("VALIDATION LOSS_____________________________")
    print(metric_dictionary["validation_loss"])
    print("TRAIN LOSS_____________________________")
    print(metric_dictionary["train_loss"])

    plt.plot(metric_dictionary["validation_loss"], 'g')
    plt.plot(metric_dictionary["train_loss"], 'r')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.savefig('loss_graphs/' + mode +'.png')

    plt.show()
