import torch
import numpy as np


class ModelCollection():
    """
    This class maintains all the different models responsible for making predictions.
    It also is responsible for routing data to the correct model in order to make
    a tfv predicition.
    """
    def __init__(self):
        self._load_models()
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }  # Dictionary for converting the days to values #
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set up the device #

    def _load_models(self):  # Load all the models #
        self._load_gru()
        self._load_tcn()
        self._load_lstm()

    def _load_gru(self):
        self._gru_model = torch.load("saved_models/gru.pth")
        self._gru_model.eval()

    def _load_tcn(self):
        self._tcn_model = torch.load("saved_models/tcn.pth")
        self._tcn_model.eval()

    def _load_lstm(self):
        self._lstm_model = torch.load("saved_models/lstm.pth")
        self._lstm_model.eval()

    def make_predicition(self, unformatted_input_data, scats_site, mode):
        """
        This functions takes the unformatted input data and then routes it. To
        the corresponding variable depending on the mode that has been selected.
        :param unformatted_input_data: A nested list that contains a nested list
        for each time step. Each nested list is of the form: [dateTimeObj for tfv, tfv]
        :param scats_site: The scats site where the predicition needs to be made.
        :param mode: The mode (model being used to make predictions) that is being
        used.
        :return:tfv prediction
        """
        tfv_prediction = None  # Stores the predicted tfv value #

        if mode == "LSTM":
            tfv_prediction = self._lstm_predict(unformatted_input_data, scats_site)
        elif mode == "GRU":
            tfv_prediction = self._gru_predict(unformatted_input_data, scats_site)
        elif mode == "TCN":
            tfv_prediction = self._tcn_predict(unformatted_input_data, scats_site)
        else:
            raise ("INVALID MODEL IN TRAFIC FLOW PREDICTOR")

        if tfv_prediction < 0:  # Removing any potential wrong negatives #
            tfv_prediction = 0

        return tfv_prediction

    def _lstm_predict(self, unformatted_input_data, scats_site):
        """
        CORRECTED: Uses simple division normalization to match training
        """
        # Extract volume values and time information
        volumes = []
        time_features = []

        for datum in unformatted_input_data:
            time_obj = datum[0]
            tfv = datum[1]

            # Calculate seconds in day for cyclical encoding
            seconds_in_day = 24 * 60 * 60
            seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

            # Cyclical time encoding
            time_sin = np.sin(2 * np.pi * seconds / seconds_in_day)
            time_cos = np.cos(2 * np.pi * seconds / seconds_in_day)

            # Cyclical weekday encoding
            weekday = time_obj.weekday()  # 0-6
            weekday_sin = np.sin(2 * np.pi * weekday / 7)
            weekday_cos = np.cos(2 * np.pi * weekday / 7)

            volumes.append(tfv)
            time_features.append([time_sin, time_cos, weekday_sin, weekday_cos])

        volumes = np.array(volumes, dtype=np.float32)
        time_features = np.array(time_features, dtype=np.float32)

        # FIXED: Use simple division normalization like training
        if hasattr(self._lstm_model, 'transform_dict'):
            max_tfv = self._lstm_model.transform_dict.get('max_tfv', 169.7)
        else:
            max_tfv = 169.7

        volumes_norm = volumes / max_tfv  # Simple division, not min-max

        # Combine features: [Volume_norm, Time_sin, Time_cos, Weekday_sin, Weekday_cos]
        input_features = np.zeros((len(volumes), 5), dtype=np.float32)
        input_features[:, 0] = volumes_norm
        input_features[:, 1:] = time_features

        # Shape for LSTM: (batch_size=1, sequence_length, features=5)
        formatted_array = np.zeros((1, input_features.shape[0], input_features.shape[1]))
        formatted_array[0] = input_features
        formatted_tensor = torch.from_numpy(formatted_array).to(torch.float32).to(self._device)

        # Run LSTM model
        yhat_norm = self._lstm_model(formatted_tensor).item()

        yhat = yhat_norm * max_tfv  # Multiply by the max tfv to get back to the original tfv #
        return yhat

    def _gru_predict(self, unformatted_input_data, scats_site):
        """
        Formats the unformatted data and inputs into the model to make a traffic flow volume prediction.
        :param unformatted_input_data: A nested list. Contains an internal list for every
        time step in the sequence. Each internal list consists of a datatime object as the
        0th element and a traffic flow volume value as the first. For example for a
        sequence length of 4 it would look something like this.
        [
        [dateTimeObj0 for tfv0, tfv0],
        [dateTimeObj1 for tfv1, tfv1],
        [dateTimeObj0 for tfv2, tfv2],
        [dateTimeObj0 for tfv3, tfv3],
        ]
        Each datTimeObj contains the time of the tfv measurement.
        :param scats_site: the scats_site where the data is from
        :return: A traffic flow volume prediction
        """
        input_data_as_list = []  # Stores the input data in the correct form #
        for datum in unformatted_input_data:  # Loop through all the time steps in the unformatted_input_data sequence #
            time_obj = datum[0]  # Get the datetime object #
            day_of_week = self._days_to_values[time_obj.strftime("%A")]  # Get the day of the week as a value. Uses the "days to values" dictionary to do this. #
            time = (time_obj.hour * 60 + time_obj.minute) / 15  # Get the time in the format expected by the model. 0 - 95 depending on the 15-minute time interval of the day that it is. #
            day_of_month = time_obj.day  # Get the day of the month 1 - 31
            tfv = datum[1]  # Get the traffic flow volume  #
            input_data_as_list.append([day_of_week, time, day_of_month, tfv])  # Append the data to the new list #

        input_data_as_np_arr = np.asarray(input_data_as_list, dtype=np.float32)  # Change the data into a numpy array #

        # Applying transformations to get the data into the right format #
        input_data_as_np_arr[:, 0] /= self._gru_model.transform_dict["max_day"]
        input_data_as_np_arr[:, 1] /= self._gru_model.transform_dict["max_time"]
        input_data_as_np_arr[:, 2] /= self._gru_model.transform_dict["max_date"]
        input_data_as_np_arr[:, 3] /= self._gru_model.transform_dict["max_tfv"]

        # Putting the data into a correctly formated tensor #
        formated_np_array = np.zeros((1, input_data_as_np_arr.shape[0], input_data_as_np_arr.shape[1]))
        formated_np_array[0] = input_data_as_np_arr
        formated_torch_tensor = torch.from_numpy(formated_np_array).to(torch.float32).to(self._device)

        # Make the prediction
        yhat = self._gru_model(formated_torch_tensor).item()
        yhat *= self._gru_model.transform_dict["max_tfv"]  # Get back to the actual tfv number #
        return yhat

    def _tcn_predict(self, unformatted_input_data, scats_site): # Same as the GRU model but with the tcn #
        """
        Formats the unformatted data and inputs into the model to make a traffic flow volume prediction.
        :param unformatted_input_data: A nested list. Contains an internal list for every
        time step in the sequence. Each internal list consists of a datatime object as the
        0th element and a traffic flow volume value as the first. For example for a
        sequence length of 4 it would look something like this.
        [
        [dateTimeObj0 for tfv0, tfv0],
        [dateTimeObj1 for tfv1, tfv1],
        [dateTimeObj0 for tfv2, tfv2],
        [dateTimeObj0 for tfv3, tfv3],
        ]
        Each datTimeObj contains the time of the tfv measurement.
        :param scats_site: the scats_site where the data is from
        :return: A traffic flow volume prediction
        """
        input_data_as_list = [] # Stores the input data in the correct form #
        for datum in unformatted_input_data:  # Loop through all the time steps in the unformatted_input_data sequence #
            time_obj = datum[0] # Get the datetime object #
            day_of_week = self._days_to_values[time_obj.strftime("%A")]  # Get the day of the week as a value. Uses the "days to values" dictionary to do this. #
            time = (time_obj.hour * 60 + time_obj.minute) / 15  # Get the time in the format expected by the model. 0 - 95 depending on the 15-minute time interval of the day that it is. #
            day_of_month = time_obj.day  # Get the day of the month 1 - 31
            tfv = datum[1]  # Get the traffic flow volume #
            input_data_as_list.append([day_of_week, time, day_of_month, tfv])  # Append the data to the new list #

        input_data_as_np_arr = np.asarray(input_data_as_list, dtype=np.float32)  # Change the data into a numpy array #

        # Applying transformations to get the data into the right format #
        input_data_as_np_arr[:, 0] /= self._tcn_model.transform_dict["max_day"]
        input_data_as_np_arr[:, 1] /= self._tcn_model.transform_dict["max_time"]
        input_data_as_np_arr[:, 2] /= self._tcn_model.transform_dict["max_date"]
        input_data_as_np_arr[:, 3] /= self._tcn_model.transform_dict["max_tfv"]

        # Putting the data into a correctly formated tensor #
        formated_np_array = np.zeros((1, input_data_as_np_arr.shape[0], input_data_as_np_arr.shape[1]))
        formated_np_array[0] = input_data_as_np_arr
        formated_torch_tensor = torch.from_numpy(formated_np_array).to(torch.float32).to(self._device)

        # Make the prediction
        yhat = self._tcn_model(formated_torch_tensor).item()
        yhat *= self._tcn_model.transform_dict["max_tfv"]  # Get back to the actual tfv number #
        return yhat