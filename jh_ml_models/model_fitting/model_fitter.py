import torch
from torch.utils.data import DataLoader
from jh_ml_models.model_fitting.data_loader import TrafficFlowDataSet
from jh_ml_models.model_code.gru_model import GRU

class Model_Fitter():
    def __init__(self, model, train_loss_function, test_loss_function, optimizer, batch_size, num_epochs, sequence_length, device, split_proportions, validate=True, save_directory=None, save_name=None):
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
        self._save_directory = save_directory
        self._save_name = save_name

    def fit_model(self, scats_site_number):
        """
        Trains and tests the model.
        :param scats_site_number: The dataloader will construct the dataset for this scats site.
        Can be set to 'all' to use all the scats site data.
        :return: A dictionary that contains the final test loss and the training and validation loss for each epoch
        """
        # Use the data loader to load the dataset
        dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                     sequence_length=self._sequence_length,
                                     selected_scats_site=scats_site_number)

        self._model.transform_dict = dataset.transform_dict  # Take the transformation dictionary from the data set so that it can be used later by the model.

        # Randomly split the dataset into the specified proportions #
        train_dataset, test_dataset, validation_dataset, _ = torch.utils.data.random_split(
            dataset,
            [self._split_proportions["train"], self._split_proportions["test"], self._split_proportions["validation"], self._split_proportions["discard"]])

        # Data loaders for different datasets. Shuffle all the data.
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self._batch_size, shuffle=True)

        # Dictionary that stores all the resulting metrics
        metric_dictionary = {
            "test_loss" : None,
            "train_loss" : [],
            "validation_loss" : [],
        }

        # Testing the untrained model
        print("Testing model that has not been trained ### ", end="")
        test_loss = self._test_model(test_loader)
        print(f"Pre-training test loss : {test_loss}")

        # Training loop
        for epoch in range(self._num_epochs):
            disp_str = f"\rEpoch {epoch + 1} #### "  # String to display the training results of the epoch. Carriage return so it appears on the same line
            train_loss = self._train_model(train_loader)  # Train the model on the current epoch
            metric_dictionary["train_loss"].append(train_loss)  # Append to the correct dictionary key #
            disp_str += f"Train loss : {train_loss} #### "  # Append onto the display string #

            if self._validate:  # If the validate flag is set to true then perform validation #
                validation_loss = self._test_model(validation_loader)  # Perform testing with the validation dataset #
                metric_dictionary["validation_loss"].append(validation_loss) # Append to the correct dictionary key #
                disp_str += f"Validation loss : {validation_loss} #### " # Append onto the display string #

            print(disp_str, end="")  # Print the display string #


        print()  # Print a line gap #
        self._test_model(test_loader)  # Test the model #
        print("Testing trained model ###", end="")
        test_loss = self._test_model(validation_loader)
        metric_dictionary["test_loss"] = test_loss
        print(f"Post-training test loss : {test_loss}")

        if self._save_directory is not None and self._save_name is not None:  # Save the model if a save path has been set
            torch.save(self._model, self._save_directory + "/" + self._save_name + ".pth")

        return metric_dictionary


    def _train_model(self, data_loader):
        """
        Use the data loader passed to train the model.
        :param data_loader: Used to load the training data
        :return: The mean loss for all the data that was trained on.
        """
        n_bat = len(data_loader)  # Get the number of batches #
        cost_sum = 0  # The sum of the costs over an epoch #
        self._model.train()  # Put the model into training mode #

        for x, y in data_loader:  # Go through each mini-batch extracting the x (input to the model) and the y (target output)
            # Send the tensor to the device GPU or CPU
            x = x.to(self._device)
            y = y.to(self._device)

            # Get the prediction made by the model and calculate the cost
            yhat = self._model(x)
            loss = self._train_loss_function(yhat, y)

            self._optimizer.zero_grad()  # Zero the accumulated gradients
            loss.backward()  # Propagate the gradients backward through the model #
            self._optimizer.step()  # Take a gradient step #

            cost_sum += loss.item()  # Accumulate the cost #

        mean_loss = cost_sum / n_bat  # Get the average cost per batch (the loss function by default gets the average over the mini-batch)
        return mean_loss

    def _test_model(self, data_loader):
        """
        Use the data loader passed to test the model.
        :param data_loader: Used to load the testing data
        :return: The mean loss for all the data that was tested on.
        """
        n_bat = len(data_loader)  # Get the number of batches #
        cost_sum = 0 # The sum of the costs over an epoch #

        self._model.eval()  # Put the model into evaluation mode #
        with torch.no_grad():  # Turn off gradients (we are evaluation we dont need to propagate gradients backward)
            for x, y in data_loader: # Go through each mini-batch extracting the x (input to the model) and the y (target output)
                # Send the tensor to the device GPU or CPU
                x = x.to(self._device)
                y = y.to(self._device)

                # Get the prediction made by the model and accumulate the cost

                yhat = self._model(x)
                cost_sum += self._test_loss_function(yhat, y).item()

        mean_loss = cost_sum/ n_bat # Get the average cost per batch (the loss function by default gets the average over the mini-batch)
        return mean_loss