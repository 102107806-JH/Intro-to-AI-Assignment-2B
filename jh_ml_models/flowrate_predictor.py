import sys
import time
from calendar import day_name
from datetime import date, datetime, timedelta
import pandas
import math
import torch
import numpy as np
import copy

class FlowratePredictor():
    def __init__(self, initial_time, sequence_length, database_file_path, mode):
        self._initial_time = initial_time
        self._sequence_length = sequence_length
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
        self._load_models()
        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()
        self._database_dictionary = self._init_data_dictionary(database_list)
        self._database_dictionary = self._remove_values_past_current_time_from_database_dict(self._database_dictionary)
        self._mode = mode
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load required models
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
        pass

    # Database handling and loading functions #
    def _init_data_dictionary(self, database_list):
        database_dictionary = {}
        #  Get all the dictionary keys #
        for entry in database_list:
            scats_site = entry[0]
            if scats_site not in database_dictionary:
                database_dictionary[scats_site] = []

        for entry in database_list:
            scats_site = entry[0]
            day_name = entry[1]
            date = entry[2]

            for time, i in enumerate(range(3, len(entry))):
                tfv = entry[i]
                database_dictionary[scats_site].append([day_name, time, date, tfv])

        return database_dictionary

    def _remove_values_past_current_time_from_database_dict(self, dictionary):
        updated_dict = {}
        for scats_site in dictionary:  # populate the updated dict with empty lists
            updated_dict[scats_site] = []

        initial_minute_rounded = math.floor(self._initial_time.minute / 15) * 15  # Drop the time to the floor of the nearest 15 min interval
        initial_hour = self._initial_time.hour
        initial_day = self._initial_time.day
        initial_month = self._initial_time.month
        initial_time_rounded = datetime(2025, initial_month, initial_day, initial_hour, initial_minute_rounded) # Times have to be less than this

        for scats_site in dictionary:
            scat_site_data_list = dictionary[scats_site]

            for entry in scat_site_data_list:
                entry_time = self._db_entry_to_time_obj(entry)

                if entry_time < initial_time_rounded:
                    updated_dict[scats_site].append(entry)

        return updated_dict

    # Database retrieval and model predictions #
    def get_data(self, time_since_initial_time, scats_site):
        query_time = self._initial_time + timedelta(hours=time_since_initial_time)
        number_of_predictions_required = self._number_of_predictions_required(query_time, scats_site)
        self._make_predictions_update_data_base(number_of_predictions_required, scats_site)
        tfv = self._query_database(query_time, scats_site)
        return tfv

    def _number_of_predictions_required(self, prediction_time, scats_site):
        final_scats_site_prediction = self._database_dictionary[scats_site][-1]
        entry_time = self._db_entry_to_time_obj(final_scats_site_prediction)
        number_predictions_required = 0

        if prediction_time > entry_time:
            seconds_dif = (prediction_time - entry_time).seconds
            minutes_dif = seconds_dif / 60
            number_predictions_required = math.floor(minutes_dif / 15)

        return number_predictions_required

    def _make_predictions_update_data_base(self, number_of_predictions_required, scats_site):

        for i in range(number_of_predictions_required):
            unformatted_input_data = copy.deepcopy(self._retrieve_model_input_sequence(scats_site))  # Data is edited inside prediction functions we must not edit the underlying database #

            tfv_prediction = None
            if self._mode == "LSTM":
                tfv_prediction = self._lstm_predict(unformatted_input_data, scats_site)
            elif self._mode == "GRU":
                tfv_prediction = self._gru_predict(unformatted_input_data, scats_site)
            elif self._mode == "TCN":
                tfv_prediction = self._tcn_predict(unformatted_input_data, scats_site)
            else:
                raise("INVALID MODEL IN TRAFIC FLOW PREDICTOR")

            self._update_database_dictionary(tfv_prediction, scats_site)

    def _retrieve_model_input_sequence(self, scats_site):
        scats_site_data = self._database_dictionary[scats_site]
        final_data_sequence = scats_site_data[-self._sequence_length :]
        return final_data_sequence

    def _lstm_predict(self, unformatted_input_data, scats_site):
        return 50

    def _gru_predict(self, unformatted_input_data, scats_site):
        for datum in unformatted_input_data:
            datum[0] = self._days_to_values[datum[0]] # Days to numbers
            datum[2] = int(datum[2][0:2])  # Date to numbers

        unformatted_input_data_as_np_arr = np.asarray(unformatted_input_data, dtype=np.float32)

        # Applying transformations
        unformatted_input_data_as_np_arr[:, 0] /= self._gru_model.transform_dict["max_day"]
        unformatted_input_data_as_np_arr[:, 1] /= self._gru_model.transform_dict["max_time"]
        unformatted_input_data_as_np_arr[:, 2] /= self._gru_model.transform_dict["max_date"]
        unformatted_input_data_as_np_arr[:, 3] /= self._gru_model.transform_dict["max_tfv"]

        formated_np_array = np.zeros((1, unformatted_input_data_as_np_arr.shape[0], unformatted_input_data_as_np_arr.shape[1]))
        formated_np_array[0] = unformatted_input_data_as_np_arr
        formated_torch_tensor = torch.from_numpy(formated_np_array).to(torch.float32).to(self._device)
        yhat = self._gru_model(formated_torch_tensor).item()
        yhat *= self._gru_model.transform_dict["max_tfv"]
        yhat = int(yhat)
        return yhat

    def _tcn_predict(self, unformatted_input_data, scats_site):
        for datum in unformatted_input_data:
            datum[0] = self._days_to_values[datum[0]]  # Days to numbers
            datum[2] = int(datum[2][0:2])  # Date to numbers

        unformatted_input_data_as_np_arr = np.asarray(unformatted_input_data, dtype=np.float32)

        # Applying transformations
        unformatted_input_data_as_np_arr[:, 0] /= self._tcn_model.transform_dict["max_day"]
        unformatted_input_data_as_np_arr[:, 1] /= self._tcn_model.transform_dict["max_time"]
        unformatted_input_data_as_np_arr[:, 2] /= self._tcn_model.transform_dict["max_date"]
        unformatted_input_data_as_np_arr[:, 3] /= self._tcn_model.transform_dict["max_tfv"]

        formated_np_array = np.zeros((1, unformatted_input_data_as_np_arr.shape[0],unformatted_input_data_as_np_arr.shape[1]))
        formated_np_array[0] = unformatted_input_data_as_np_arr
        formated_torch_tensor = torch.from_numpy(formated_np_array).to(torch.float32).to(self._device)
        yhat = self._tcn_model(formated_torch_tensor).item()
        yhat *= self._tcn_model.transform_dict["max_tfv"]
        yhat = int(yhat)
        return yhat

    def _update_database_dictionary(self, tfv_prediction, scats_site):
        final_scats_site_prediction = self._database_dictionary[scats_site][-1]
        entry_time = self._db_entry_to_time_obj(final_scats_site_prediction)
        new_entry_time = entry_time + timedelta(minutes=15)
        new_entry = self._time_obj_to_db_entry(new_entry_time)
        new_entry.append(tfv_prediction)
        self._database_dictionary[scats_site].append(new_entry)

    def _query_database(self, query_time, scats_site):
        scats_site_data = self._database_dictionary[scats_site]

        for entry in reversed(scats_site_data):
            bottom_time = self._db_entry_to_time_obj(entry)
            top_time = bottom_time + timedelta(minutes=15)

            if query_time >= bottom_time and top_time > query_time:
                tfv = entry[-1]
                return tfv
        raise("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")

    # Helper methods #
    def _db_entry_to_time_obj(self, entry):
        temp_time = entry[1]
        total_minutes = temp_time * 15
        entry_minutes = total_minutes % 60
        entry_hours = math.floor(total_minutes / 60)
        entry_day = int(entry[2][0:2])
        entry_month = int(entry[2][3:5])
        entry_time = datetime(2025, entry_month, entry_day, entry_hours, entry_minutes)
        return entry_time

    def _time_obj_to_db_entry(self, time_obj):
        day_name = time_obj.strftime("%A")
        hour = time_obj.hour
        minute = time_obj.minute
        entry_time = int(hour * 4 + minute / 15)
        entry_date = time_obj.strftime("%d/%m/%Y")
        return [day_name, entry_time, entry_date]



