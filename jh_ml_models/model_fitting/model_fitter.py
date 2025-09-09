import torch
from torch.utils.data import DataLoader
from jh_ml_models.model_fitting.data_loader import TrafficFlowDataSet

class Model_Fitter():
    def __init__(self, model, train_loss_function, test_loss_function, optimizer, batch_size, num_epochs, sequence_length, device, model_save_path=None):
        self._model = model
        self._train_loss_function = train_loss_function
        self._test_loss_function = test_loss_function
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._sequence_length = sequence_length
        self._device = device
        self._model_save_path = model_save_path

    def fit_model(self, scats_site_number):
        dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                     sequence_length=self._sequence_length,
                                     selected_scats_site=scats_site_number)


        self._model.transform_dict = dataset.transform_dict

        train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset,[0.9, 0.05, 0.05])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self._batch_size, shuffle=True)

        print("Initial Test __________")
        self._test_model(test_loader)
        print()

        for epoch in range(self._num_epochs):
            print(f"Epoch {epoch + 1}\n-------------")
            self._train_model(validation_loader)
            self._test_model(test_loader)
            print()

        if self._model_save_path is not None:
            torch.save(self._model, self._model_save_path)


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
        print(f"Train Loss: {mean_loss}")

    def _test_model(self, data_loader):
        n_bat = len(data_loader)
        cost_sum = 0

        self._model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                yhat = self._model(x)
                cost_sum += self._test_loss_function (yhat, y).item()

        mean_loss = cost_sum/ n_bat
        print(f"Test Loss (MAE): {mean_loss}")