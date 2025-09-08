import torch

class ModelCollection():
    def __init__(self):
        pass

        # MODEL LOADING ###########################################################
        def _load_models(self):
            self._load_gru()
            self._load_tcn()
            self._load_lstm()

        def _load_gru(self):
            self._gru_model = torch.load("saved_models/gru.pth")
            self._gru_model.eval()
            pass

        def _load_tcn(self):
            self._tcn_model = torch.load("saved_models/tcn.pth")
            self._tcn_model.eval()

        def _load_lstm(self):
            """
            This is where you load your lstm. Feel free to add any attributes that
            you may need when making your predictions later.
            """
            pass
