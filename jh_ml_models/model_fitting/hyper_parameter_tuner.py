import copy
import math
import random
import torch
from torch import nn
from jh_ml_models.model_code.gru_model import GRU
from jh_ml_models.model_code.tcn_model import TCN
from jh_ml_models.model_fitting.model_fitter import Model_Fitter
from datetime import datetime, timedelta
class HyperParameterTuner():
    def __init__(self, mode, epochs_per_run):
        self._mode = mode
        self._sequence_length = 12
        self._feature_size = 4
        self._epochs_per_run = epochs_per_run
        self._split_proportions = {
            "train": 0.01,
            "test": 0.2,
            "validation": 0.005
        }
        self._split_proportions["discard"] = 1 - self._split_proportions["train"] - self._split_proportions["test"] - self._split_proportions["validation"]
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def random_search(self, hyper_parameter_dictionary, finish_time):
        best_loss = float("inf")
        best_loss_hyper_parameter_datum_dictionary  = None
        print(f"Hyper parameter tuning will stop at approximately: {finish_time.strftime('%d/%m/%Y  %H:%M:%S')}")
        while datetime.now() < finish_time:
            print()
            hyper_parameter_datum_dictionary = {
                # Common hps
                "lr": hyper_parameter_dictionary["lr"].get_random_bound_val(),
                "batch_size": hyper_parameter_dictionary["batch_size"].get_random_bound_val(),
                # GRU
                "hidden_size": hyper_parameter_dictionary["hidden_size"].get_random_bound_val(),
                "num_layers": hyper_parameter_dictionary["num_layers"].get_random_bound_val(),
                # TCN
                "kernel_size": hyper_parameter_dictionary["kernel_size"].get_random_bound_val(),
                "C1_out_channels": hyper_parameter_dictionary["C1_out_channels"].get_random_bound_val()
            }
            try:
                loss = self._test_hp_datum(hyper_parameter_datum_dictionary)
            except:
                print("Invalid hyper parameter combination. Continuing to next random sample")
                continue
            if loss < best_loss:
                best_loss = loss
                best_loss_hyper_parameter_datum_dictionary = copy.deepcopy(hyper_parameter_datum_dictionary)
        print("RESULTS--------------------------------------")
        print(f"Best Loss {best_loss}")
        print(f"Hyper-Parameters")
        print_dict(best_loss_hyper_parameter_datum_dictionary)

    def _test_hp_datum(self, hyper_parameter_datum_dictionary):
        metric_dictionary = None
        if self._mode == "gru":
            metric_dictionary = self._test_hp_datum_gru(hyper_parameter_datum_dictionary)
        elif self._mode == "tcn":
            metric_dictionary = self._test_hp_datum_tcn(hyper_parameter_datum_dictionary)
        return metric_dictionary["test_loss"]

    def _test_hp_datum_gru(self, hyper_parameter_datum_dictionary):
        print("---------------------------------------------------------------")
        print(f"GRU HYPER-PARAMETER DATUM TEST. The settings are:"
              f"\nLearning Rate: {hyper_parameter_datum_dictionary['lr']}"
              f"\nBatch Size: {hyper_parameter_datum_dictionary['batch_size']}"
              f"\nHidden Size: {hyper_parameter_datum_dictionary['hidden_size']}"
              f"\nNum Layers: {hyper_parameter_datum_dictionary['num_layers']}")

        model = GRU(
            feature_size=self._feature_size,
            sequence_length=self._sequence_length,
            hidden_size=hyper_parameter_datum_dictionary["hidden_size"],
            num_layers=hyper_parameter_datum_dictionary["num_layers"],
            device=self._device).to(self._device)

        train_loss_function = nn.MSELoss()
        test_loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter_datum_dictionary["lr"])
        scats_site_number = 'ALL'

        fitter = Model_Fitter(model=model,
                              train_loss_function=train_loss_function,
                              test_loss_function=test_loss_function,
                              optimizer=optimizer,
                              batch_size=hyper_parameter_datum_dictionary["batch_size"],
                              num_epochs=self._epochs_per_run,
                              sequence_length=self._sequence_length,
                              device=self._device,
                              split_proportions=self._split_proportions,
                              validate=True,
                              model_save_path=None)
        metric_dictionary = fitter.fit_model(scats_site_number)
        return metric_dictionary

    def _test_hp_datum_tcn(self, hyper_parameter_datum_dictionary):
        print("---------------------------------------------------------------")
        print(f"TCN HYPER-PARAMETER DATUM TEST. The settings are:"
              f"\nLearning Rate: {hyper_parameter_datum_dictionary['lr']}"
              f"\nBatch Size: {hyper_parameter_datum_dictionary['batch_size']}"
              f"\nKernel Size: {hyper_parameter_datum_dictionary['kernel_size']}"
              f"\nNumber of Layer 1 output channels: {hyper_parameter_datum_dictionary['C1_out_channels']}")

        model = TCN(
            feature_size=self._feature_size,
            sequence_length=self._sequence_length,
            kernel_size=hyper_parameter_datum_dictionary["kernel_size"],
            c1_out_channels=hyper_parameter_datum_dictionary["C1_out_channels"],
            device=self._device).to(self._device)

        train_loss_function = nn.MSELoss()
        test_loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter_datum_dictionary["lr"])
        scats_site_number = 'ALL'

        fitter = Model_Fitter(model=model,
                              train_loss_function=train_loss_function,
                              test_loss_function=test_loss_function,
                              optimizer=optimizer,
                              batch_size=hyper_parameter_datum_dictionary["batch_size"],
                              num_epochs=self._epochs_per_run,
                              sequence_length=self._sequence_length,
                              device=self._device,
                              split_proportions=self._split_proportions,
                              validate=True,
                              model_save_path=None)
        metric_dictionary = fitter.fit_model(scats_site_number)
        return metric_dictionary

class HyperParameter():
    def __init__(self, lower_limit, upper_limit, data_type, data_list=None):
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._data_type = data_type
        self._data_list = data_list

    def get_random_bound_val(self):
        if self._data_type == int:
            return random.randint(self._lower_limit, self._upper_limit)
        elif self._data_type == float:
            return random.random() * (self._upper_limit - self._lower_limit) + self._lower_limit
        elif self._data_type == list:
            return self._data_list[random.randint(0, len(self._data_list)-1)]

        raise Exception("Type not supported")


def print_dict(dict):
    for key in dict:
        print(f"{key} : {dict[key]}")


