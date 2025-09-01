import torch
from torch.utils.data import DataLoader
from jh_ml_models.data_loader import TrafficFlowDataSet

class Model_Fitter():
    def __init__(self, model, train_loss_function, test_loss_function, optimizer, batch_size, num_epochs, device):
        self._model = model
        self._train_loss_function = train_loss_function
        self._test_loss_function = test_loss_function
        self._optimizer = optimizer
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._device = device

    def fit_model(self, scats_site_number):
        dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                     sequence_length=3,
                                     selected_scats_site=scats_site_number)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset,[0.8, 0.2])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=True)

        print("Untrained Test __________")
        self._test_model(test_loader)
        print()

        for epoch in range(self._num_epochs):
            print(f"Epoch {epoch}\n-------------")
            self._train_model(train_loader)
            self._test_model(test_loader)
            print()

    def _train_model(self, data_loader):
        num_batches = len(data_loader)
        total_loss = 0
        self._model.train()

        for x, y in data_loader:
            x = x.to(self._device)
            y = y.to(self._device)

            output = self._model(x)
            loss = self._train_loss_function(output, y)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train Loss: {avg_loss}")

    def _test_model(self, data_loader):

        num_batches = len(data_loader)
        total_loss = 0

        self._model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self._device)
                y = y.to(self._device)

                output = self._model(x)
                total_loss += self._test_loss_function (output, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test Loss (MAE): {avg_loss}")