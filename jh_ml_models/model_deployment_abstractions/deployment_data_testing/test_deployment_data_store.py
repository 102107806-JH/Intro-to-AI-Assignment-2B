import pandas
from datetime import date, datetime, timedelta
import math

class TestDeploymentDataStore():
    """
    This class stores the deployment data and is used by the ModelTester. Please
    read the 'current_deployment_data_store.py' file before this as this will make
    more sense
    """
    def __init__(self, database_file_path):
        self._test_data_dictionary = self._load_test_data(database_file_path)  # Store the data from the fake database in a dictionary #

    def _load_test_data(self, database_file_path):
        """
        This function loads the data out of the fake database file and puts it
        into a dictionary.
        """
        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()  # Convert the db file into a list #
        current_data_dictionary = {}  # Dictionary that will store the current data #

        for entry in database_list:  # Loop through all entries in the data_base list #
            scats_site = entry[0]  # Get the scats site number #

            if scats_site not in current_data_dictionary:  # If the scat site is not a key in the dictionary add it and set its value to an empty list #
                current_data_dictionary[scats_site] = []

            date = entry[2]  # Get the date of the entry in the list #

            for time, i in enumerate(range(3, len(entry))):  # Go through entry values in the list
                tfv = entry[i]  # Get the traffic flow volume #
                # Create a datetime object as the entry
                entry_time = datetime(year=int(date[6:]),
                                      month=int(date[3:5]),
                                      day=int(date[:2]),
                                      hour=math.floor(time * 15 / 60),
                                      minute=time * 15 % 60)

                current_data_dictionary[scats_site].append([entry_time, tfv])

        return current_data_dictionary  # Return the current data dictionary #

    def query(self, query_time, scats_site):
        """
         Returns the tfv value from the current data for the inputted query time
         and scats site number.
        """
        scats_site_data = self._test_data_dictionary[scats_site]  # Get a reference to the scats_site data #
        for entry in scats_site_data:  # Go through all the scats site data #
            bottom_time = entry[0]  # The time stored in the entry #
            top_time = bottom_time + timedelta(minutes=15)  # 15 minutes past the bottom time #

            # If the query time lies between the two then the entry has been found #
            if query_time >= bottom_time and top_time > query_time:
                tfv = entry[-1]
                return tfv

        raise ("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")

    def query_sequence(self, query_time, scats_site, sequence_length):
        """
        Get the sequence for the 'query_time' and 'scats_site'.
        """
        scats_site_data = self._test_data_dictionary[scats_site]  # Get a reference to the scats_site data #
        for i, entry in enumerate(scats_site_data):  # Go through all the scats site data #
            bottom_time = entry[0]  # The time stored in the entry #
            top_time = bottom_time + timedelta(minutes=15)  # 15 minutes past the bottom time #
            # If the query time lies between the two then the entry has been found #
            if query_time >= bottom_time and top_time > query_time:
                sequence = scats_site_data[i-sequence_length+1: i + 1]  # have to shift the operation forward 1 to extract the current time #
                return sequence

        raise ("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")

