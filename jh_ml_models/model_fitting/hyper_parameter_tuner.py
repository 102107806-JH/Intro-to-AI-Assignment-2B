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
        self._mode = mode  # The mode ie: the type of model that we will be training #
        self._sequence_length = 12  # The sequence length number of time steps #
        self._feature_size = 4  # The number of features per time step #
        self._epochs_per_run = epochs_per_run  # The number of epochs per testing run of the hyperparameters #
        # The proportion of the overall dataset that are used for different splits. Was only a small amount
        # of the original dataset in order to speed up the process of hyperparameter selection
        self._split_proportions = {
            "train": 0.01,
            "test": 0.2,
            "validation": 0.005
        }
        self._split_proportions["discard"] = 1 - self._split_proportions["train"] - self._split_proportions["test"] - self._split_proportions["validation"]  # Proportion of the dataset that will be discarded
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # The device that is used #

    def random_search(self, hyper_parameter_dictionary, finish_time):
        """
        Implements a random search of the hyperparameter space.
        :param hyper_parameter_dictionary: A dictionary containing custom hyperparameter objects. Each key is a hyperparameter
        and the values are the objects. The object just provides a convenient way to access the hyperparameter objects.
        :param finish_time: A datetime object when you wish the training to stop. Once this time is passed the current
        hyperparameter data point be evaluated will be finished then the program will end,
        :return: None
        """
        best_loss = float("inf")  # The best cost that has been encountered so far #
        best_loss_hyper_parameter_datum_dictionary  = None  # Will store the dictionary that contains the hyperparameters that correspond to the best cost #
        print(f"Hyper parameter tuning will stop at approximately: {finish_time.strftime('%d/%m/%Y  %H:%M:%S')}")  # Display to the user approximately when the program will end #
        while datetime.now() < finish_time:  # Keep going until we are past the final time #
            print()
            # This dictionary that displays the specific hyperparameter datum to be tested
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
            # Incase there is an invalid hyperparameter selection that is made.
            try:
                loss = self._test_hp_datum(hyper_parameter_datum_dictionary)
            except:
                print("Invalid hyper parameter combination. Continuing to next random sample")
                continue

            # Update the relevant variables when a better loss is encountered
            if loss < best_loss:
                best_loss = loss
                best_loss_hyper_parameter_datum_dictionary = copy.deepcopy(hyper_parameter_datum_dictionary)

        # Print the results
        print("RESULTS--------------------------------------")
        print(f"Best Loss {best_loss}")
        print(f"Hyper-Parameters")
        print_dict(best_loss_hyper_parameter_datum_dictionary)

    def _test_hp_datum(self, hyper_parameter_datum_dictionary):
        """
        Passes the hyperparameter dictionary to the function with the correct model inside
        :param hyper_parameter_datum_dictionary:  Hyperparameter datum that is use to
        train the model.
        :return: The test loss after the model has been trained.
        """
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


