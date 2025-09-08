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
        time_dif = end_datetime - start_datetime
        time_dif_total_seconds = time_dif.days * 24 * 60 * 60 + time_dif.seconds
        number_of_predictions = + math.floor((time_dif_total_seconds / 60) / 15)
        output_dictionary = {
            "Prediction_Times": [],
            "GRU": [],
            "TCN": [],
            "LSTM": [],
            "Targets": []
        }
        current_predicition_datetime = start_datetime
        for i in range(number_of_predictions):
            sequence = self._test_deployment_data_store.query_sequence(
                query_time=current_predicition_datetime - timedelta(minutes=15 * prediction_depth),
                scats_site=scats_site,
                sequence_length=sequence_length)

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

            target = self._test_deployment_data_store.query(
                query_time=current_predicition_datetime,
                scats_site=scats_site)

            output_dictionary["GRU"].append(gru_prediction)
            output_dictionary["TCN"].append(tcn_prediction)
            output_dictionary["LSTM"].append(lstm_prediction)
            output_dictionary["Targets"].append(target)

            current_predicition_datetime = current_predicition_datetime + timedelta(minutes=15)
        return output_dictionary

