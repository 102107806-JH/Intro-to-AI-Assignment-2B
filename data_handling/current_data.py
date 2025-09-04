import pandas
from datetime import date, datetime, timedelta
import math

class CurrentData():
    def __init__(self, initial_time, database_file_path):
        """
        This class uses a dictionary to store the current available data. The
        dictionary maintains keys to all the scats sites. The values are the
        data that is available at a given site. Further more only data upto the
        current time is loaded into the dictionary. This is to simulate only having
        past data available.
        """
        self._current_data_dictionary = self._load_data_into_data_base(database_file_path, initial_time)

    def _load_data_into_data_base(self, database_file_path, initial_time):
        """
        This function reads data into the current data dictionary from the
        provided database excel file. Additionally it only loads in entries that
        are before the current time.
        """
        # Round down the initial time to the nearest 15 minute interval
        initial_time_rounded_down = datetime(year=initial_time.year,
                                             month=initial_time.month,
                                             day=initial_time.day,
                                             hour=initial_time.hour,
                                             minute=initial_time.minute - initial_time.minute % 15)

        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()  # Convert the db file into a list #
        current_data_dictionary = {}  # Dictionary that will store the current data #

        for entry in database_list:  # Loop through all entries in the data_base list #
            scats_site = entry[0]  # Get the scats site number #

            if scats_site not in current_data_dictionary:  # If the scat site is not a key in the dictionary add it and set its value to an empty list #
                current_data_dictionary[scats_site] = []

            date = entry[2] # Get the date of the entry in the list #

            for time, i in enumerate(range(3, len(entry))):  # Go through entry values in the list
                tfv = entry[i]  # Get the traffic flow volume #
                # Create a datetime object as the entry
                entry_time = datetime(year=int(date[6:]),
                                    month=int(date[3:5]),
                                    day=int(date[:2]),
                                    hour=math.floor(time * 15 / 60),
                                    minute=time * 15 % 60)
                # We only add the entry if it has happened before the initial time. The entries come in
                # 15 minute intervals. So say for example we had an initial time of
                # year = 2025, month = 8, day = 5, hour = 5, minute = 39. This would
                # first be rounded down to hour = 5, minute = 30 (at the start
                # of the function). The condition below means that the last entry
                # added would be hour = 5, minute = 15. This is done because
                # the traffic flow at hour = 5, minute = 30 is the traffic flow
                # from minute = 30 (inclusive) to minute = 45 (exclusive).
                if initial_time_rounded_down > entry_time:
                    current_data_dictionary[scats_site].append([entry_time, tfv])

        return current_data_dictionary  # Return the current data dictionary #

    def get_scats_data(self, scats_site):
        return self._current_data_dictionary[scats_site]  # Get the nested list that contains all of the data for a given scats site

    def get_input_sequence(self, scats_site, sequence_length):
        """
        Get the nested list that is going to form the input sequence for the
        model.
        """
        scats_site_data = self._current_data_dictionary[scats_site]
        final_data_sequence = scats_site_data[-sequence_length:]
        return final_data_sequence

    def append_data_to_scats_site(self, scats_site, data):
        self._current_data_dictionary[scats_site].append(data)  # Append data into the data dictionary

    def query(self, query_time, scats_site):
        """
         Returns the tfv value from the current data for the inputted query time
         and scats site number.
        """
        scats_site_data = self._current_data_dictionary[scats_site]  # Get a reference to the scats_site data #
        for entry in reversed(scats_site_data):  # Go through all the scats site data #
            bottom_time = entry[0]  # The time stored in the entry #
            top_time = bottom_time + timedelta(minutes=15)  # 15 minutes past the bottom time #

            # If the query time lies between the two then the entry has been found #
            if query_time >= bottom_time and top_time > query_time:
                tfv = entry[-1]
                return tfv

        raise("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")
