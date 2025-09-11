import pandas
from datetime import date, datetime, timedelta
import math


class CurrentDeploymentDataStore():
    """
    In the realworld we would have access to a database that would enable retrival
    of upto date data. However in this task we do not have access to that. As such
    this has to be simulated. This is what this class achieves. It loads the data
    from 01/07/2025 00:00 - x/08/2025 xx:xx  for all the scats sites into a dictionary.
    Here the 'x' represents a quantity that is determined by the initial time.
    Basically it uses the data for the 08/2025 and pretends that it is the current
    data. The reason that the 7th month was used was in case data near the start of
    8th month was requested and this then made the sequence go back into the 7th month.
    Ensuring that the initial time is in the 8th month is done outside this class.
    Furthermore it allows appending of data to the deployment data dictionary. Because there are times
    when this is required because future data points may be required. For example
    suppose we are traversing through the graph it may take more than 1 timestep to get
    to a given node as such the model would need to predict all the timesteps required
    to get to that node. These predictions need to be able to be stored to allow the model
    to use them for the successively deeper predictions and additionally once they are
    stored they dont have to be re-calculated.
    """
    def __init__(self, initial_time, database_file_path):
        self._current_deployment_data_dictionary = self._load_deployment_data(database_file_path, initial_time)

    def _load_deployment_data(self, database_file_path, initial_time):
        """
        This function reads data into the current data dictionary from the
        provided database Excel file. Additionally, it only loads in entries that
        are before the 'initial time'.
        :param database_file_path: The path to the fake Excel database file. That
        contains the data that would be used during deployment.
        :param initial_time: The initial time at which the pathfinding will start.
        The deployment data which is after this will not be loaded.
        :return: A data dictionary with all the current deployment data
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
        return self._current_deployment_data_dictionary[scats_site]  # Get the nested list that contains all of the data for a given scats site

    def get_input_sequence(self, scats_site, sequence_length):
        """
        Get the nested list that is going to form the input sequence for the
        model.
        :param scats_site: The scats site where we want to extract the sequence.
        :param sequence_length: The length of the sequence that we want to extract.
        :return: Nested list where each nested entry is time step data of a given
        teme step for the scats site. Returns the last possible sequence.
        """
        scats_site_data = self._current_deployment_data_dictionary[scats_site]  # Get the data list for the scats site #
        final_data_sequence = scats_site_data[-sequence_length:]  #  Get the last possible sequence #
        return final_data_sequence

    def append_data_to_scats_site(self, scats_site, data):
        self._current_deployment_data_dictionary[scats_site].append(data)  # Append data into the data dictionary #

    def query(self, query_time, scats_site):
        """
        Returns the tfv value from the current data for the inputted query time
         and scats site number.
        :param query_time: The time of the tfv value you want to obtain.
        :param scats_site: The scats site of the tfv value that you want to obtain.
        :return: tfv value for the given scats site and the query time.
        """
        scats_site_data = self._current_deployment_data_dictionary[scats_site]  # Get a reference to the scats_site data #
        for entry in reversed(scats_site_data):  # Go through all the scats site data #
            bottom_time = entry[0]  # The time stored in the entry #
            top_time = bottom_time + timedelta(minutes=15)  # 15 minutes past the bottom time #

            # If the query time lies between the two then the entry has been found #
            if query_time >= bottom_time and top_time > query_time:
                tfv = entry[-1]
                return tfv
        #raise Exception("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")
        return 999 # In case this is reached punish the entry
