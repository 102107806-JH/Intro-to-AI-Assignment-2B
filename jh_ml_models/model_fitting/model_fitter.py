import torch
from torch.utils.data import DataLoader
from jh_ml_models.model_fitting.data_loader import TrafficFlowDataSet

class Model_Fitter():
    def __init__(self, model, train_loss_function, test_loss_function, optimizer, batch_size, num_epochs, sequence_length, device, split_proportions, validate=True, model_save_path=None):
        self._model = model
        self._train_loss_function = train_loss_function
        self._test_loss_function = test_loss_function
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._sequence_length = sequence_length
        self._device = device
        self._split_proportions = split_proportions
        self._validate = validate
        self._model_save_path = model_save_path

    def fit_model(self, scats_site_number):
        dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                     sequence_length=self._sequence_length,
                                     selected_scats_site=scats_site_number)

        self._model.transform_dict = dataset.transform_dict

        train_dataset, test_dataset, validation_dataset, _ = torch.utils.data.random_split(
            dataset,
            [self._split_proportions["train"], self._split_proportions["test"], self._split_proportions["validation"], self._split_proportions["discard"]])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self._batch_size, shuffle=True)

        metric_dictionary = {
            "test_loss" : None,
            "train_loss" : [],
            "validation_loss" : [],
        }

        print("Testing model that has not been trained ### ", end="")
        test_loss = self._test_model(test_loader)
        print(f"Pre-training test loss : {test_loss}")

        for epoch in range(self._num_epochs):
            disp_str = f"\rEpoch {epoch + 1} #### "
            train_loss = self._train_model(train_loader)
            metric_dictionary["train_loss"].append(train_loss)
            disp_str += f"Train loss : {train_loss} #### "

            if self._validate:
                validation_loss = self._test_model(validation_loader)
                metric_dictionary["validation_loss"].append(validation_loss)
                disp_str += f"Validation loss : {validation_loss} #### "

            print(disp_str, end="")


        print()
        self._test_model(test_loader)
        print("Testing trained model ###", end="")
        test_loss = self._test_model(validation_loader)
        metric_dictionary["test_loss"] = test_loss
        print(f"Post-training test loss : {test_loss}")

        if self._model_save_path is not None:
            torch.save(self._model, self._model_save_path)

        return metric_dictionary


    def _train_model(self, data_loader):
        n_bat = len(data_loader)
        cost_sum = 0
        self._model.train()

        for x, y in data_loader:
            x = x.to(self._device)
            y = y.to(self._device)

            yhat = self._model(x)
            loss = self._train_loss_function(yhat, y)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            cost_sum += loss.item()

        mean_loss = cost_sum / n_bat
        return mean_loss

    def _test_model(self, data_loader):
        n_bat = len(data_loader)
        cost_sum = 0

        self._model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                yhat = self._model(x)
                cost_sum += self._test_loss_function(yhat, y).item()

        mean_loss = cost_sum/ n_bat
        return mean_loss