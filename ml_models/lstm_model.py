import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class LstmTrafficModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=1, output_size=1):  # These are set in main.py
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        batch_size = input_seq.size(0)  # Input Size
        self.hidden_state = self.init_hidden(batch_size)
        lstm_out, self.hidden_state = self.lstm(input_seq, self.hidden_state)  # Returns
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions.view(-1, 1)  # Output Size

class LstmFlowRatePredictor:
    def __init__(self, model, min_val, max_val):
        self._model = model
        self._min_val = min_val
        self._max_val = max_val
        self._model.eval()

    def _inverse_normalize(self, value):
        # Converts normalized values back to original state after ML training for Search Algorithm
        return value * (self._max_val - self._min_val) + self._min_val

    def make_prediction(self, input_sequence):
        normalized_input = (input_sequence - self._min_val) / (self._max_val - self._min_val)
        input_tensor = torch.tensor(normalized_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():  # Edge Case: To avoid backpropagation through the model
            prediction = self._model(input_tensor)
        return self._inverse_normalize(prediction.item())

def train_model(train_loader, min_val, max_val, hyperparameters):
    model = LstmTrafficModel(
        hidden_layer_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers']
    )
    loss_function = nn.MSELoss()
    optimizer_map = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'Adamax': optim.Adamax,
        'NAdam': optim.NAdam,
        'NAG': optim.SGD
    }

    optimizer_name = hyperparameters.get('optimizer', 'Adam')
    if optimizer_name in optimizer_map:
        optimizer_class = optimizer_map[optimizer_name]  # Get the optimizer class from the map
        # Instantiate the optimizer with the model parameters and learning rate
        optimizer = optimizer_class(model.parameters(), lr=hyperparameters['learning_rate'])
    else:
        raise ValueError(f"Invalid optimizer name {optimizer_name}.")

    for epoch in tqdm(range(hyperparameters['epochs']), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

    model.eval()
    total_correct = 0
    total_predictions = 0
    with torch.no_grad():
        # Wrap the DataLoader with tqdm to show progress during evaluation
        for seq, labels in tqdm(train_loader, desc="Evaluating Model"):
            y_pred = model(seq)
            y_pred_denormalized = y_pred * (max_val - min_val) + min_val
            labels_denormalized = labels * (max_val - min_val) + min_val
            epsilon = 1e-6  # Edge Case: Small epsilon to prevent division by zero
            correct_predictions = torch.abs(y_pred_denormalized - labels_denormalized) / (labels_denormalized + epsilon) < 0.05
            total_correct += correct_predictions.sum().item()
            total_predictions += labels.size(0)
    accuracy = (total_correct / total_predictions) * 100
    return LstmFlowRatePredictor(model, min_val, max_val), accuracy