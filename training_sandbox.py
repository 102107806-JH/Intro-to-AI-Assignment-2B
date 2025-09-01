from jh_ml_models.gru_model_runner import gru_model_runner
from torch.utils.data import DataLoader
from jh_ml_models.gru_model import GRU
import numpy as np
from torch import nn
from jh_ml_models.test_and_train import train_model, test_model


if __name__ == "__main__":

    gru_model_runner()