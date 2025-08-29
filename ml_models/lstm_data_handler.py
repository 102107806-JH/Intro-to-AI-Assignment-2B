import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_sequences(data, lookback):
    X, y = [], []  # Initialize empty lists for input and target sequences
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def preprocess_data(file_path, scats_number, lookback=12, batch_size=32):
    # Load the data using pandas.read_excel for .xlsx files
    df = pd.read_excel(file_path)
    df_scats = df[df['SCATS_Number'] == scats_number].iloc[:, 3:] # Skip Date and Weekday columns
    traffic_data = df_scats.values.flatten().astype(float)    # Flatten the data into a single time series

    # Normalize the data into 0-1 range because the output most be positive for Dijkstar to work
    min_val = traffic_data.min()
    max_val = traffic_data.max()
    normalized_data = (traffic_data - min_val) / (max_val - min_val)

    # Create sequences
    X, y = create_sequences(normalized_data, lookback)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    # Create DataLoader object for training
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, min_val, max_val