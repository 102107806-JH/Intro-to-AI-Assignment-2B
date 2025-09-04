import sys
import time
from calendar import day_name
from datetime import date, datetime, timedelta
import pandas
import math

class FlowratePredictor():
    def __init__(self, initial_time, sequence_length, database_file_path):
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
        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()
        self._database_dictionary = self._init_data_dictionary(database_list)
        self._database_dictionary = self._remove_values_past_current_time_from_database_dict(self._database_dictionary)
        self._mode = "GRU"

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
            unformatted_input_data = self._retrieve_model_input_sequence(scats_site)

            tfv_prediction = None
            if self._mode == "LSTM":
                tfv_prediction = self._lstm_predict(unformatted_input_data, scats_site)
            elif self._mode == "GRU":
                tfv_prediction = self._gru_predict(unformatted_input_data, scats_site)
            elif self._mode == "TCN":
                tfv_prediction = self._tcn_predict(unformatted_input_data, scats_site)

            self._update_database_dictionary(tfv_prediction, scats_site)

    def _retrieve_model_input_sequence(self, scats_site):
        scats_site_data = self._database_dictionary[scats_site]
        final_data_sequence = scats_site_data[-self._sequence_length :]
        return final_data_sequence

    def _lstm_predict(self, unformatted_input_data, scats_site):
        return 50

    def _gru_predict(self, unformatted_input_data, scats_site):
        return 50

    def _tcn_predict(self, unformatted_input_data, scats_site):
        return 50

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



    def _load_models(self):
        pass

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



