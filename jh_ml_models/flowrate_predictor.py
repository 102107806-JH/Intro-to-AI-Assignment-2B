import time
from datetime import date, datetime, timedelta

class FlowratePredictor():
    def __init__(self):
        self._initial_time = None

    def set_initial_time(self, initial_time):
        self._initial_time = initial_time

    def _load_models(self):
        pass

    def make_prediction(self, elapsed_time):
        prediction_time = self._initial_time + timedelta(hours=elapsed_time)

        return 0

