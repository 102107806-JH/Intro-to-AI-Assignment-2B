import torch
import numpy as np
import copy
from datetime import datetime, timedelta

class FlowratePredictionTester():
    def __init__(self):
        self._load_models()  # Load models #
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

    def _load_models(self):
        self._load_gru()
        self._load_tcn()
        self._load_lstm()

    def _load_gru(self):
        self._gru_model = torch.load("saved_models/gru.pth")
        self._gru_model.eval()
        pass

    def _load_tcn(self):
        self._tcn_model = torch.load("saved_models/tcn.pth")
        self._tcn_model.eval()

    def _load_lstm(self):
        """
        This is where you load your lstm. Feel free to add any attributes that
        you may need when making your predictions later.
        """
        pass

    def test(self, unformatted_input_data, scats_site, mode, prediction_depth):
        unformatted_input_data = copy.deepcopy(unformatted_input_data) # Data may be edited inside prediction functions we must not edit the underlying data #

        tfv_prediction = None
        for i in range(prediction_depth):

            if mode == "LSTM":
                tfv_prediction = self._lstm_predict(unformatted_input_data, scats_site)
            elif mode == "GRU":
                tfv_prediction = self._gru_predict(unformatted_input_data, scats_site)
            elif mode == "TCN":
                tfv_prediction = self._tcn_predict(unformatted_input_data, scats_site)
            else:
                raise("INVALID MODEL IN TRAFIC FLOW PREDICTOR")

            if tfv_prediction < 0: # Removing any potential wrong negatives #
                tfv_prediction = 0

            final_time = unformatted_input_data[-1][0]

            unformatted_input_data.pop(0)
            unformatted_input_data.append([final_time + timedelta(minutes=15), tfv_prediction])

        return tfv_prediction

    def _lstm_predict(self, unformatted_input_data, scats_site):
        """
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

        You will have to format the data yourself. Please refer to the _gru_predict
        for inspiration if needed.
        """
        return None

    def _gru_predict(self, unformatted_input_data, scats_site):
        """
        Formats the unformatted data and inputs into the model to make a traffic flow volume prediction.
        """
        input_data_as_list = []
        for datum in unformatted_input_data:  # Put the text based datum into numbers #
            time_obj = datum[0]
            day_of_week = self._days_to_values[time_obj.strftime("%A")]
            time = (time_obj.hour * 60 + time_obj.minute) / 15
            day_of_month = time_obj.day
            tfv = datum[1]
            input_data_as_list.append([day_of_week, time, day_of_month, tfv])

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
        #yhat = int(yhat)  # If Desired the prediction does not need to be converted to an integer #
        return yhat

    def _tcn_predict(self, unformatted_input_data, scats_site): # Same as the GRU model but with the tcn #
        """
        Formats the unformatted data and inputs into the model to make a traffic flow volume prediction.
        """
        input_data_as_list = []
        for datum in unformatted_input_data:  # Put the text based datum into numbers #
            time_obj = datum[0]
            day_of_week = self._days_to_values[time_obj.strftime("%A")]
            time = (time_obj.hour * 60 + time_obj.minute) / 15
            day_of_month = time_obj.day
            tfv = datum[1]
            input_data_as_list.append([day_of_week, time, day_of_month, tfv])

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
        # yhat = int(yhat)  # If Desired the prediction does not need to be converted to an integer #
        return yhat