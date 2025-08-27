# Fake model that makes a dummy prediction will be replaced with an actual trained model
class MockModel:
    def __init__(self):
        pass

    def make_prediction(self):
        return 1