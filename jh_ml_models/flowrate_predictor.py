import time
from datetime import date, datetime, timedelta

class FlowratePredictor():
    def __init__(self, initial_time, sequence_length, database_file):
        self._initial_time = initial_time
        self._sequence_length = sequence_length
        self._database_file = database_file

    def _load_required_db_data(self):
        pass

    def _load_models(self):
        pass

    def make_prediction(self, elapsed_time, scats_site):
        prediction_time = self._initial_time + timedelta(hours=elapsed_time)

        return 0

