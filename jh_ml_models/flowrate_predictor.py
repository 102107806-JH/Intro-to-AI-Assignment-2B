from data_base.mock_data_base import CurrentData
from datetime import date, datetime, timedelta
import math
import copy
import torch
import numpy as np

class FlowratePredictor():
    def __init__(self, initial_time, sequence_length, database_file_path, mode):
        self._initial_time = initial_time
        self._sequence_length = sequence_length
        self._current_data = CurrentData(initial_time=initial_time,
                                        database_file_path=database_file_path)
        self._load_models()
        self._mode = mode
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }  # Dictionary for converting the days to values #
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 1 Load required models
    def _load_models(self):
        self._load_gru()
        self._load_tcn()
        self._load_lstm()

    def _load_gru(self):
        self._gru_model = torch.load("saved_models/gru/gru.pth")
        self._gru_model.eval()

    def _load_tcn(self):
        self._tcn_model = torch.load("saved_models/gru/gru.pth")
        self._tcn_model.eval()

    def _load_lstm(self):
        """
        This is where you load your lstm. Feel free to add any attributes that
        you may need when making your predictions.
        """
        pass

    def get_data(self, time_since_initial_time, scats_site):
        query_time = self._initial_time + timedelta(hours=time_since_initial_time)  # The time of query to the database #
        number_predictions_required = self._number_of_predictions_required(query_time, scats_site)
        self._make_predictions_and_update_current_data(number_predictions_required, scats_site)
        tfv = self._current_data.query(query_time, scats_site)
        return tfv

    def _number_of_predictions_required(self, prediction_time, scats_site):
        final_scats_site_entry = self._current_data.get_scats_data(scats_site)[-1]  # Get the final scats site entry for the given scat site #
        entry_time = final_scats_site_entry[0]  # Get the time that this entry was made as a time object #
        number_predictions_required = 0  # The number of predictions that will be required #

        if prediction_time > entry_time: # Predictions are only needed if the prediction time is greater than the time of the final entry #
            seconds_dif = (prediction_time - entry_time).seconds  # Get the difference between prediction and the entry time #
            minutes_dif = seconds_dif / 60  # Get the number of minutes difference #
            number_predictions_required = math.floor(minutes_dif / 15)  # Predictions are valid for 15 minutes #

        return number_predictions_required  # Return the number of predictions that are required #

    def _make_predictions_and_update_current_data(self, number_of_predictions_required, scats_site):

        for i in range(number_of_predictions_required):
            unformatted_input_data = copy.deepcopy(self._current_data.get_input_sequence(scats_site, self._sequence_length))  # Data is edited inside prediction functions we must not edit the underlying database #

            tfv_prediction = None
            if self._mode == "LSTM":
                tfv_prediction = self._lstm_predict(unformatted_input_data, scats_site)
            elif self._mode == "GRU":
                tfv_prediction = self._gru_predict(unformatted_input_data, scats_site)
            elif self._mode == "TCN":
                tfv_prediction = self._tcn_predict(unformatted_input_data, scats_site)
            else:
                raise("INVALID MODEL IN TRAFIC FLOW PREDICTOR")

            if tfv_prediction < 0: # Removing any potential wrong negatives #
                tfv_prediction = 0

            date_obj_to_be_appended = unformatted_input_data[-1][0] + timedelta(minutes=15)
            data_to_be_appended_to_current_data = [date_obj_to_be_appended, tfv_prediction]
            self._current_data.append_data_to_scats_site(scats_site, data_to_be_appended_to_current_data) # The new prediction data needs to be added to the database dictionary #

    def _lstm_predict(self, unformatted_input_data, scats_site):
        return None

    def _gru_predict(self, unformatted_input_data, scats_site):
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
        Refer to '_gru_predict'
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
