import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LstmTrafficModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, num_layers=1, output_size=1):
        """"
        LSTM model definitions and hyperparameters.

        :param
        input_size: int, number of input features.
        hidden_layer_size: int, number of hidden units in the LSTM layer.
        num_layers: int, number of LSTM layers.
        output_size: int, number

        :return:
        Trained model statistics
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)  # processes sequences of input
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        """
        :param
            input_sequence:
             torch.tensor
             shape (batch_size seq_len input_size)

        :return:
            torch.tensor
            shape (batch_size output_size)
        """
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])  # last timestep output
        return predictions.view(-1, 1)


class LstmFlowRatePredictor:
    def __init__(self, model, min_val, max_val, device):
        """
        A wrapper class that keeps the model and hyperparameters for each training session.
        :param
            model: LstmTrafficModel
            min_val: float, minimum value of the input sequence.
            max_val: float, maximum value of the input sequence.
            device: torch.device, device to run the model on.
        """
        self._model = model
        self._min_val = min_val
        self._max_val = max_val
        self._device = device
        self._model.eval()

    def _inverse_normalize(self, value):
        """
        Rescale model output back to the original volume units. (min_val, max_val)
        """
        return value * (self._max_val - self._min_val) + self._min_val

    def make_prediction(self, input_sequence):
        """
        Used to make a single step prediction
        :param
            input_sequence:
            torch.tensor
            shape (batch_size seq_len input_size)

        :return
                float, predicted flow rate
       """
        input_seq = input_sequence.copy()
        input_seq[:, 0] = (input_seq[:, 0] - self._min_val) / (self._max_val - self._min_val + 1e-8)  # Normalize the
        # traffic volume feature before passing to model.
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self._device)  # Convert to tensor,
        # and add batching dimension.
        with torch.no_grad():  # Edge Case: Run model forward pass without tracking gradients
            prediction = self._model(input_tensor)
        return self._inverse_normalize(prediction.item())  # Return prediction in original units


def train_model(train_loader, min_val, max_val, hyperparameters, patience=5):
    """
    Trains the model on the training data (model_data.xlsx) then returns the trained model and the sMAPE (the margin of
    error calculated as a percentage) So 9% means the data can predict accurately 91% of the time.

    :param
        train_loader: torch.utils.data.DataLoader
        train_loader: torch.utils.data.DataLoader, the training data loader.
        min_val: float, minimum value of the input sequence.
        max_val: float, maximum value of the input sequence.
        hyperparameters: dict, hyperparameters for the model.
        patience: int, patience for early stopping.

    :return
        LstmFlowRatePredictor, sMAPE score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if CUDA supported GPU is available
    # for training if not then default to the CPU.

    # Initialise the model with some sample hyperparameters which are defined in main.py
    model = LstmTrafficModel(
        input_size=5,
        hidden_layer_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers']
    ).to(device)

    # Loss function = Mean Squared Error
    loss_function = nn.MSELoss()

    optimizer_map = {  # Used for Finding Hyperparameters when training the model using the hyperparameter tuning script
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'Adamax': optim.Adamax,
        'NAdam': optim.NAdam,
        'NAG': optim.SGD
    }
    optimizer = optimizer_map[hyperparameters['optimizer']](model.parameters(), lr=hyperparameters['learning_rate'])

    # Variables for early stopping and patience procedures
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in tqdm(range(hyperparameters['epochs']), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (best loss: {best_loss:.4f})")
                break
    if best_model_state:  # Restore the best model weights with the lowest validation loss
        model.load_state_dict(best_model_state)

    # Evaluating the Model on the test data
    model.eval()
    mae_total, smape_total = 0.0, 0.0
    with torch.no_grad():  # Edge Case: Run model forward pass without tracking gradients
        for seq, labels in tqdm(train_loader, desc="Evaluating Model"):
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            y_pred_denorm = y_pred * (max_val - min_val) + min_val  # Undo normalization to compute error in
            # original scale
            labels_denorm = labels * (max_val - min_val) + min_val

            # MAE calculation
            mae_total += torch.sum(torch.abs(y_pred_denorm - labels_denorm)).item()

            # sMAPE calculation
            smape_total += torch.sum(
                torch.abs(y_pred_denorm - labels_denorm) /
                ((torch.abs(y_pred_denorm) + torch.abs(labels_denorm)) / 2 + 1e-6)
            ).item()
    # Normalize errors by dataset size
    mae = mae_total / len(train_loader.dataset)
    smape = (smape_total / len(train_loader.dataset)) * 100  # Calculate percentage of errors
    print(f"Evaluation Results -> MAE: {mae:.2f}, sMAPE: {smape:.2f}%")

    return LstmFlowRatePredictor(model, min_val, max_val, device), smape