from prototype_model_tester.flowrate_prediction_tester import FlowratePredictionTester
from prototype_model_tester.test_data_store import TestDataStore


class ModelTester():
    def __init__(self):
        test_data_store = TestDataStore(database_file_path="data/data_base.xlsx")
        flowrate_prediction_tester = FlowratePredictionTester()
