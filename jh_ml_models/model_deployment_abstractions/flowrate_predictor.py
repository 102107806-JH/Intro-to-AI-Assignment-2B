from jh_ml_models.model_deployment_abstractions.current_deployment_data_store import CurrentDeploymentDataStore
from datetime import timedelta
import math
import copy
import torch
import numpy as np
from jh_ml_models.model_code.model_collection import ModelCollection

class FlowratePredictor():
    """
    This class is used by the problem class to develop flow rate predictions.
    It works with the CurrentDeploymentDataStore class to store predicted values and
    retrieve training data.It wraps the model collection and feeds it data in
    order to make a prediction.
    """
    def __init__(self, initial_time, sequence_length, database_file_path, mode):

        self._initial_time = initial_time
        self._sequence_length = sequence_length
        self._current_data = CurrentDeploymentDataStore(
            initial_time=initial_time,
            database_file_path=database_file_path)  # Init an instance of the current data object #

        self._mode = mode  # Determines what model should be used #
        self._model_collection = ModelCollection()  # Contains the collection all the models is used to make predictions #

    def get_data(self, time_since_initial_time, scats_site):
        """
        This function is a wrapper that joins all the sub functions together.
        First it gets the time of the query that is being made. It then determines
        based on the scat site and the time of the query the number of predictions
        that need to be made. It then makes the predictions and updates the
        self._current_data object with these predictions so that they can be
        used later in the algorithm if required. Lastly it retrieves the data
        from the current data which corresponds to the current query time and
        scats site.
        """
        query_time = self._initial_time + timedelta(hours=time_since_initial_time)  # The time of query to the database #
        number_predictions_required = self._number_of_predictions_required(query_time, scats_site)
        self._make_predictions_and_update_current_data(number_predictions_required, scats_site)
        tfv = self._current_data.query(query_time, scats_site)  # Need to query because it is possible that all predictions have already been made #
        return tfv

    def _number_of_predictions_required(self, prediction_time, scats_site):
        """
        For the given prediction time and scats_site this function calculates
        the number of predictions that need to be made. This needs to be done
        because it is possible that a query could be made multiple time steps
        past the last data point for a scats site.
        """
        final_scats_site_entry = self._current_data.get_scats_data(scats_site)[-1]  # Get the final scats site entry for the given scat site #
        entry_time = final_scats_site_entry[0]  # Get the time that this entry was made as a time object #
        number_predictions_required = 0  # The number of predictions that will be required #

        if prediction_time > entry_time: # Predictions are only needed if the prediction time is greater than the time of the final entry #
            seconds_dif = (prediction_time - entry_time).seconds  # Get the difference between prediction and the entry time #
            minutes_dif = seconds_dif / 60  # Get the number of minutes difference #
            number_predictions_required = math.floor(minutes_dif / 15)  # Predictions are valid for 15 minutes #

        return number_predictions_required  # Return the number of predictions that are required #

    def _make_predictions_and_update_current_data(self, number_of_predictions_required, scats_site):
        """
        This function loops over the number of predictions that need to be made
        and makes them. It retrieves the data needed to make the prediction it
        then routes this data to the model collection which predicts the tfv.
        Lastly it calls the self._current_data objects append function and adds
        the prediction made to the current data. This is done because it is possible
        the same scats_site at the same time will be accessed again later. This
        saves compute as each scat_site at each time interval only needs to be
        predicted once.
        """
        for i in range(number_of_predictions_required):
            unformatted_input_data = copy.deepcopy(self._current_data.get_input_sequence(scats_site, self._sequence_length))  # Data  edited inside prediction functions we must not edit the underlying database #

            tfv_prediction = self._model_collection.make_predicition(unformatted_input_data, scats_site, self._mode)

            date_obj_to_be_appended = unformatted_input_data[-1][0] + timedelta(minutes=15)
            data_to_be_appended_to_current_data = [date_obj_to_be_appended, tfv_prediction]
            self._current_data.append_data_to_scats_site(scats_site, data_to_be_appended_to_current_data) # The new prediction data needs to be added to the database dictionary #
