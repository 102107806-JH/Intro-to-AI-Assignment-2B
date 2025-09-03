import sys
import time
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
        self._remove_values_past_current_time_from_database_dict()
        print("end")

    def _remove_values_past_current_time_from_database_dict(self):
        updated_dict = {}
        for scats in self._database_dictionary:  # populate the updated dict with empty lists
            updated_dict[scats] = []

        max_minute = math.floor(self._initial_time.minute / 15) * 15  # Drop the time to the floor of the nearest 15 min interval
        max_hour = self._initial_time.hour
        max_day = self._initial_time.day
        max_month = 8
        max_time = datetime(2025, max_month, max_day, max_hour, max_minute) # Times have to be less than this

        for scats in self._database_dictionary:
            scat_site_data_list = self._database_dictionary[scats]

            for entry in scat_site_data_list:
                temp_time = entry[1]
                total_minutes = temp_time * 15

                entry_minutes = total_minutes % 60
                entry_hours = math.floor(total_minutes / 60)
                entry_day = int(entry[2][0:2])
                entry_month = int(entry[2][3:5])
                entry_time = datetime(2025, entry_month, entry_day,entry_hours, entry_minutes)

                if entry_time < max_time:
                    updated_dict[scats].append(entry)
                    print(entry)

        print()



    def _init_data_dictionary(self, database_list):
        database_dictionary = {}
        #  Get all the dictionary keys
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

    def _load_models(self):
        pass

    def _retrieve_data(self, time, scats_site):
        pass

    def make_prediction(self, time_since_initial_time, scats_site):
        prediction_time = self._initial_time + timedelta(hours=time_since_initial_time)




        return 0

