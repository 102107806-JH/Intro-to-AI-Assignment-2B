from jh_ml_models.model_deployment_abstractions.deployment_data_testing.deployment_data_model_tester import DeploymentDataModelTester

class MultiScatSiteDeploymentDataModelTester:
    def __init__(self, scat_site_list):
        """
        :param scat_site_list: A list containing all the scat sites that want to be tested
        """
        self._scat_site_list = scat_site_list
        self._model_tester = DeploymentDataModelTester(database_file_path="data/data_base.xlsx")

    def test_models(self, start_time, end_time, predicition_depth, sequence_length):
        """

        :param start_time:The first predicition time inclusive
        :param end_time: The final predicition time exclusive
        :param predicition_depth: How far ahead the model has to predict
        :param sequence_length: The length of the sequence that the model can process
        :return: A nested dictionary where the first key returns the dictionary for that
        scat site and the second which specific results you want to get from that dictionary
        """

        results = {} # A dictionary that contains all the results dictionary for all the scats sites
        for scat_site in self._scat_site_list:
            print(f"Testing scat site {scat_site}")
            # Store the results for the scat site in the dictionary
            results[scat_site] = self._model_tester.test_models(
                scats_site=scat_site,
                prediction_depth=predicition_depth,
                sequence_length=sequence_length,
                start_datetime=start_time,
                end_datetime=end_time)

        return results

