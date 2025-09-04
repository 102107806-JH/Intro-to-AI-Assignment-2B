import pandas
from datetime import date, datetime, timedelta
import math

class CurrentData():
    def __init__(self, initial_time, database_file_path):
        self._current_data_dictionary = self._load_data_into_data_base(database_file_path, initial_time)

    def _load_data_into_data_base(self, database_file_path, initial_time):
        initial_time_rounded_down = datetime(year=initial_time.year,
                                             month=initial_time.month,
                                             day=initial_time.day,
                                             hour=initial_time.hour,
                                             minute=initial_time.minute - initial_time.minute % 15)

        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()
        current_data_dictionary = {}  # Dictionary that will act as the database

        # Create a key for each scats site and set the value to an empty list #
        for entry in database_list:
            scats_site = entry[0]

            if scats_site not in current_data_dictionary:
                current_data_dictionary[scats_site] = []

            date = entry[2]

            for time, i in enumerate(range(3, len(entry))):
                tfv = entry[i]
                entry_time = datetime(year=int(date[6:]),
                                    month=int(date[3:5]),
                                    day=int(date[:2]),
                                    hour=math.floor(time * 15 / 60),
                                    minute=time * 15 % 60)
                if initial_time_rounded_down > entry_time:
                    current_data_dictionary[scats_site].append([entry_time, tfv])


        return current_data_dictionary  # Return the created database dictionary #

    def get_scats_data(self, scats_site):
        return self._current_data_dictionary[scats_site]

    def get_input_sequence(self, scats_site, sequence_length):
        scats_site_data = self._current_data_dictionary[scats_site]
        final_data_sequence = scats_site_data[-sequence_length:]
        return final_data_sequence

    def append_data_to_scats_site(self, scats_site, data):
        self._current_data_dictionary[scats_site].append(data)

    def query(self, query_time, scats_site):
        scats_site_data = self._current_data_dictionary[scats_site]

        for entry in reversed(scats_site_data):  # Start searching in reverse #
            bottom_time = entry[0]  # Get the time of the given entry #
            top_time = bottom_time + timedelta(minutes=15)  # Get the maximum time for a query to be considered part of this prediction #

            if query_time >= bottom_time and top_time > query_time:  # If the query time is between the two times then we have found the correct query #
                tfv = entry[-1]  # The tfv is the final entry for the given entry #
                return tfv
        raise("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")  # This should never happen #
