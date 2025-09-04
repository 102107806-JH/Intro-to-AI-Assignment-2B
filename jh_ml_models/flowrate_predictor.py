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
        """
        Begin by initializing required variables. Following this the prediction
        models are loaded. Then the 'database' is loaded. We dont have access to
        a real database in this task so the 'database' is just an excel file with
        all the required data. This data is first placed into a list. The data
        is then loaded into a dictionary that has a key for each scat site and
        the values are nested lists containing all the data for that scat site
        in chronological order. The database contains values from
        01/07/2025 - 31/08/2025. The data in the 08 month is treated as the
        current data. The reason the 07 month was included was in case a query
        near the start of the month was made which due to querys "looking back"
        could have resulted in an error. When a query is made it will draw the
        data from the database on the 8th month.
        """
        self._initial_time = initial_time  # This is the time that the search was started at #
        self._sequence_length = sequence_length  # The sequence length (number of timesteps) of the sequences that are to be fed into the model #
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }  # Dictionary for converting the days to values #
        self._mode = mode # This is the mode / model that the predictions will be made with #
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Get the device which the

        self._load_models()  # Load the time series prediction models #
        database_list = (pandas.read_excel(database_file_path).to_numpy()).tolist()  # Extract the data from the mock database file #
        self._database_dictionary = self._init_data_dictionary(database_list)  # Move the data from a list into a dictionary #
        self._database_dictionary = self._remove_values_past_current_time_from_database_dict(self._database_dictionary)  # Remove values past the initial time to simulate a real situation where we would not have future values #


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

    # Database handling and loading functions #############################################################################
    def _init_data_dictionary(self, database_list):
        """
        This function transfers the data between the database_list and
        the data dictionary. In the process it formats the data into the
        correct form. Each key is a scat site and the values are the data for
        that scat site in the form of a nested list. For example the value
        stored at scat_site = 970 would look something like the following
        [
        ['Monday', 0, 1, 98],
        ['Monday', 1, 1, 58],
        ['Monday', 2, 1, 78],
        ...
        ]
        """
        database_dictionary = {}  # Dictionary that will act as the database

        # Create a key for each scats site and set the value to an empty list #
        for entry in database_list:
            scats_site = entry[0]
            if scats_site not in database_dictionary:
                database_dictionary[scats_site] = []

        # Transfer the information from the database list into the dictionary. And format the data eg ['Tuesday', 4, '02/07/2025', 95]
        for entry in database_list:
            scats_site = entry[0]
            day_name = entry[1]
            date = entry[2]

            for time, i in enumerate(range(3, len(entry))):
                tfv = entry[i]
                database_dictionary[scats_site].append([day_name, time, date, tfv])

        return database_dictionary  # Return the created database dictionary #

    def _remove_values_past_current_time_from_database_dict(self, dictionary):
        """
        Values need to be removed from the database dictionary. This is because
        (as mentioned) we are treating the 8th month as the current month because
        we dont have access to realtime data. As such the database contains
        data that is in the future. This is not allowed and this data needs
        to be deleted. This is what this function does. It deletes all invalid
        data.
        """
        updated_dict = {}  # Updated dictionary that will contain the updated data #
        for scats_site in dictionary:  # Populate the updated dict with empty lists for each unique scats site #
            updated_dict[scats_site] = []

        initial_minute_rounded = math.floor(self._initial_time.minute / 15) * 15  # Drop the time to the floor of the nearest 15 min interval
        initial_hour = self._initial_time.hour  # Get the initial hour #
        initial_day = self._initial_time.day  # Get the initial day #
        initial_month = self._initial_time.month  # Get the initial month. When the initial time is generated this is set to 8 #
        initial_time_rounded = datetime(2025, initial_month, initial_day, initial_hour, initial_minute_rounded) # The initial time rounded down to the closest 15 min interval #

        for scats_site in dictionary:  # Loop through all keys (scats sites) in the dictionary #
            scat_site_data_list = dictionary[scats_site]  # Get the data for each scats site key #

            for entry in scat_site_data_list:  # Loop through all the entries for a given scats site #
                entry_time = self._db_entry_to_time_obj(entry)  # Get the entry time of the scats entry #

                if entry_time < initial_time_rounded:  # Only allow add times where the entry time is less than the initial time rounded. This ensures that we will be predicting the first time #
                    updated_dict[scats_site].append(entry)

        return updated_dict  # Return the updated dictionary #

    # Database retrieval and model predictions ##############################################################################
    def get_data(self, time_since_initial_time, scats_site):
        """
        This function wraps all the database retrieval and model prediction functions.
        First it gets the time of the query. It does this by adding the time
        since the initial time to the initial time. It then gets the number of
        predictions that need to be made. This is done because as the
        graph is traversed it is possible that the query time may lay a few
        time steps ahead of the last retrievable time in the database. For
        example say I am going from node A -> node B -> node C. The time predicted
        time taken to get to node B from A could be 45 minutes and as such when
        it comes time to make the prediction of the time to get to node C multiple
        time steps may need to be predicted. After this the mock database is
        updated with the predictions that have been made so that they dont
        need to be made again. Finally, after these predictions have been made
        and stored in the database is queried to retrieve the tfv
        value for the given query time and scats site.
        """
        query_time = self._initial_time + timedelta(hours=time_since_initial_time)  # The time of query to the database #
        number_of_predictions_required = self._number_of_predictions_required(query_time, scats_site)  # Calculates the number of predictions that need to be made for the query time and the given scats site. It could be more than one as travel time progresses #
        self._make_predictions_update_data_base(number_of_predictions_required, scats_site)  # Make the required predictions and update the database so that these predictions dont have to be re-made #
        tfv = self._query_database(query_time, scats_site)  # Query the updated database which will allow us to get the the tfv for the scats_site at the querry time #
        return tfv

    def _number_of_predictions_required(self, prediction_time, scats_site):
        """
        This function gets the number of predictions that need to be made for
        a given time and scats site. We can retrieve the last entry for a scats
        site and the time this occurred this can be compared with the current
        time to get the number of entries that are required.
        """
        final_scats_site_entry = self._database_dictionary[scats_site][-1]  # Get the final scats site entry for the given scat site #
        entry_time = self._db_entry_to_time_obj(final_scats_site_entry)  # Get the time that this entry was made as a time object #
        number_predictions_required = 0  # The number of predictions that will be required #

        if prediction_time > entry_time: # Predictions are only needed if the prediction time is greater than the time of the final entry #
            seconds_dif = (prediction_time - entry_time).seconds  # Get the difference between prediction and the entry time #
            minutes_dif = seconds_dif / 60  # Get the number of minutes difference #
            number_predictions_required = math.floor(minutes_dif / 15)  # Predictions are valid for 15 minutes #

        return number_predictions_required  # Return the number of predictions that are required #

    def _make_predictions_update_data_base(self, number_of_predictions_required, scats_site):
        """
        This function loops through the number of required predictions that
        need to be made and then has the selected model perform these predictions.
        It then updates the database with the predictions that have been made.
        So that they don't need to be made again later.

        """

        for i in range(number_of_predictions_required):  # Loop through the number of required predictions
            # Bellow function retrieves the data the models are required to make the predictions on #
            unformatted_input_data = copy.deepcopy(self._retrieve_model_input_sequence(scats_site))  # Data is edited inside prediction functions we must not edit the underlying database #

            # Feed the unformated data into functions who's responsibility is to make the predictions #
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

            self._update_database_dictionary(tfv_prediction, scats_site)  # The new prediction data needs to be added to the database dictionary #

    def _retrieve_model_input_sequence(self, scats_site):
        """
        This function takes a scat site an retrieves the input sequence that
        the models will use for inference. It has been determined how many
        predictions need to be this function is just called to retrieve the
        data for one of those predictions.
        """
        scats_site_data = self._database_dictionary[scats_site]  # Get all the data for the given scat site #
        final_data_sequence = scats_site_data[-self._sequence_length :]  # Get the sequence required to make the prediction
        return final_data_sequence  # Return the sequence in list format

    def _lstm_predict(self, unformatted_input_data, scats_site):
        """
        Takes the unformatted input data WHICH IS IN A LIST and the scats site number.
        The list is in the form [sequence][features]. Each feature is in the form
        [day_of_week, time, date, tfv]. For example for a sequence of length 3
        it could look like:
        [
        ['Tuesday', 45, 23, 345],
        ['Tuesday', 46, 23, 345],
        ['Tuesday', 47, 23, 345]
        ]
        You will need to get this data into the correct form for your model to
        make predictions I would recommend. Changing the mode in the
        'path_finding_demo.py' file to LSTM and using the debugger to inspect the
        'unnformated_input_data'. Also please observe the '_gru_predict' and 'tcn_predict'
        functions if your are at all unclear or for ideas. I have included the
        scats site as a parameter in case you need it.
        """
        return None

    def _gru_predict(self, unformatted_input_data, scats_site):
        """
                Takes the unformatted input data WHICH IS IN A LIST and the scats site number.
                The list is in the form [sequence][features]. Each feature is in the form
                [day_of_week, time, date, tfv]. For example for a sequence of length 3
                it could look like:
                [
                ['Tuesday', 45, 23, 345],
                ['Tuesday', 46, 23, 345],
                ['Tuesday', 47, 23, 345]
                ]
        """
        for datum in unformatted_input_data:  # Put the text based datum into numbers #
            datum[0] = self._days_to_values[datum[0]] # Days to numbers
            datum[2] = int(datum[2][0:2])  # Date to numbers

        unformatted_input_data_as_np_arr = np.asarray(unformatted_input_data, dtype=np.float32)  # Change the data into a numpy array #

        # Applying transformations to get the data into the right format #
        unformatted_input_data_as_np_arr[:, 0] /= self._gru_model.transform_dict["max_day"]
        unformatted_input_data_as_np_arr[:, 1] /= self._gru_model.transform_dict["max_time"]
        unformatted_input_data_as_np_arr[:, 2] /= self._gru_model.transform_dict["max_date"]
        unformatted_input_data_as_np_arr[:, 3] /= self._gru_model.transform_dict["max_tfv"]

        # Putting the data into a correctly formated tensor #
        formated_np_array = np.zeros((1, unformatted_input_data_as_np_arr.shape[0], unformatted_input_data_as_np_arr.shape[1]))
        formated_np_array[0] = unformatted_input_data_as_np_arr
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
        for datum in unformatted_input_data:
            datum[0] = self._days_to_values[datum[0]]
            datum[2] = int(datum[2][0:2])

        unformatted_input_data_as_np_arr = np.asarray(unformatted_input_data, dtype=np.float32)

        unformatted_input_data_as_np_arr[:, 0] /= self._tcn_model.transform_dict["max_day"]
        unformatted_input_data_as_np_arr[:, 1] /= self._tcn_model.transform_dict["max_time"]
        unformatted_input_data_as_np_arr[:, 2] /= self._tcn_model.transform_dict["max_date"]
        unformatted_input_data_as_np_arr[:, 3] /= self._tcn_model.transform_dict["max_tfv"]

        formated_np_array = np.zeros((1, unformatted_input_data_as_np_arr.shape[0],unformatted_input_data_as_np_arr.shape[1]))
        formated_np_array[0] = unformatted_input_data_as_np_arr
        formated_torch_tensor = torch.from_numpy(formated_np_array).to(torch.float32).to(self._device)
        yhat = self._tcn_model(formated_torch_tensor).item()
        yhat *= self._tcn_model.transform_dict["max_tfv"]
        #yhat = int(yhat)
        return yhat

    def _update_database_dictionary(self, tfv_prediction, scats_site):
        """
        Appends a prediction to the database at the corresponding scats site.
        This means that the prediction does not need to be made again.
        """
        final_scats_site_entry = self._database_dictionary[scats_site][-1]  # Get the final database entry #
        entry_time = self._db_entry_to_time_obj(final_scats_site_entry)  # Get the time of the final entry as datetime object #
        new_entry_time = entry_time + timedelta(minutes=15)  # The next entry is recorded 15 minutes in the future #
        new_entry = self._time_obj_to_db_entry(new_entry_time, tfv_prediction)  # Get the new entry time in the correct list format #
        self._database_dictionary[scats_site].append(new_entry)  # Append the entire new entry into the corresponding scats site list in the dictionary #

    def _query_database(self, query_time, scats_site):
        """
        Searches the database for a given scat site and retreives the entry at
        the inputted query time.
        """
        scats_site_data = self._database_dictionary[scats_site]  # Get the data for the given scats site #

        for entry in reversed(scats_site_data):  # Start searching in reverse #
            bottom_time = self._db_entry_to_time_obj(entry)  # Get the time of the given entry #
            top_time = bottom_time + timedelta(minutes=15)  # Get the maximum time for a query to be considered part of this prediction #

            if query_time >= bottom_time and top_time > query_time:  # If the query time is between the two times then we have found the correct query #
                tfv = entry[-1]  # The tfv is the final entry for the given entry #
                return tfv
        raise("QUERY TIME FOR SCATS SITE HAS NO CORRESPONDING ENTRY!")  # This should never happen #

    # Helper methods ##############################################################################
    def _db_entry_to_time_obj(self, entry):
        """
        Formats a database entry into a python datetime object
        """
        temp_time = entry[1]
        total_minutes = temp_time * 15
        entry_minutes = total_minutes % 60
        entry_hours = math.floor(total_minutes / 60)
        entry_day = int(entry[2][0:2])
        entry_month = int(entry[2][3:5])
        entry_time = datetime(2025, entry_month, entry_day, entry_hours, entry_minutes)
        return entry_time

    def _time_obj_to_db_entry(self, time_obj, tfv):
        """
        Formats a datetime object and tfv value into a database entry list.
        """
        day_name = time_obj.strftime("%A")
        hour = time_obj.hour
        minute = time_obj.minute
        entry_time = int(hour * 4 + minute / 15)
        entry_date = time_obj.strftime("%d/%m/%Y")
        return [day_name, entry_time, entry_date, tfv]



