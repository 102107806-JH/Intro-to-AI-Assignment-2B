from jh_ml_models.model_deployment_abstractions.deployment_data_testing.deployment_data_model_tester import DeploymentDataModelTester

class MultiScatSiteDeploymentDataModelTester:
    def __init__(self, scat_site_list):
        self._scat_site_list = scat_site_list
        self._model_tester = DeploymentDataModelTester(database_file_path="data/data_base.xlsx")

    def test_models(self, start_time, end_time, predicition_depth, sequence_length):
        start_time = start_time.replace(month=8)
        end_time = end_time.replace(month=8)

        results = {} # A dictionary that contains all the results dictionary for all the scats sites
        for scat_site in self._scat_site_list:
            print(f"Test scat site {scat_site}")
            results[scat_site] = self._model_tester.test_models(
                scats_site=scat_site,
                prediction_depth=predicition_depth,
                sequence_length=sequence_length,
                start_datetime=start_time,
                end_datetime=end_time)

        return results

