from jh_ml_models.model_deployment_abstractions.deployment_data_testing.flowrate_prediction_tester import FlowratePredictionTester
from jh_ml_models.model_deployment_abstractions.deployment_data_testing.test_deployment_data_store import TestDeploymentDataStore
import math
from datetime import timedelta

class DeploymentDataModelTester():
    """
    Wraps the 'TestDeploymentDataStore' and the 'FlowratePredictionTester' and
    enables testing of the models on the deployment data.
    """
    def __init__(self, database_file_path):
        self._test_deployment_data_store = TestDeploymentDataStore(database_file_path=database_file_path)
        self._flowrate_prediction_tester = FlowratePredictionTester()

    def test_models(self, scats_site, prediction_depth, sequence_length, start_datetime, end_datetime):
        """
        :param scats_site: The scats_site where you want the testing to be performed.
        :param prediction_depth: How many time steps ahead the model is trying to predict.
        Previous predictions are used to predict further and further depths.
        :param sequence_length: The number of time steps in the input sequence.
        :param start_datetime: datetime object that stores when you want the first
        predicition to occur should be between 01/08/2025 - 31/08/2025.
        :param end_datetime: The date of the final prediction.
        :return: A dictionary that contains information regarding the outputs of
        the models over different timesteps.
        """
        time_dif = end_datetime - start_datetime  # The time difference between the start and the endtime #
        time_dif_total_seconds = time_dif.days * 24 * 60 * 60 + time_dif.seconds  # The time difference in seconds #
        number_of_predictions = math.floor((time_dif_total_seconds / 60) / 15)  # The number of predictions that need to be made. Get the total minutes with the "/60" then get the number of 15 minute chunks with the "/15"
        # Dictionary of lists that will store the output lists #
        output_dictionary = {
            "Prediction_Times": [],
            "GRU": [],
            "TCN": [],
            "LSTM": [],
            "Targets": []
        }
        current_predicition_datetime = start_datetime  # The time of the current predicition that is used within the for loop #
        for i in range(number_of_predictions):
            # Get the sequence to make the prediction
            sequence = self._test_deployment_data_store.query_sequence(
                query_time=current_predicition_datetime - timedelta(minutes=15 * prediction_depth),
                scats_site=scats_site,
                sequence_length=sequence_length)
            # Model predictions
            gru_prediction = self._flowrate_prediction_tester.test(
                unformatted_input_data=sequence,
                scats_site=scats_site,
                mode="GRU",
                prediction_depth=prediction_depth)

            tcn_prediction = self._flowrate_prediction_tester.test(
                unformatted_input_data=sequence,
                scats_site=scats_site,
                mode="TCN",
                prediction_depth=prediction_depth)

            lstm_prediction = self._flowrate_prediction_tester.test(
                unformatted_input_data=sequence,
                scats_site=scats_site,
                mode="LSTM",
                prediction_depth=prediction_depth)
            # Get the target so that it can be compared to the model predicitions
            target = self._test_deployment_data_store.query(
                query_time=current_predicition_datetime,
                scats_site=scats_site)

            # Append predictions to the output dictionary list
            output_dictionary["GRU"].append(gru_prediction)
            output_dictionary["TCN"].append(tcn_prediction)
            output_dictionary["LSTM"].append(lstm_prediction)
            output_dictionary["Targets"].append(target)
            output_dictionary["Prediction_Times"].append(current_predicition_datetime - timedelta(minutes=15 * prediction_depth))

            current_predicition_datetime = current_predicition_datetime + timedelta(minutes=15)  # Go forward 15 minutes to retrieve the next prediction
        return output_dictionary

