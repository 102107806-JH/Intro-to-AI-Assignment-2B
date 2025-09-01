import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_sequences(data, lookback):
    """
    In this function we create input-output sequence pairs from the data for time series forcasting.

    :param lookback:
        1. data (np.array): full feature matrix of shape (T, num_features), where T = number of time steps.
        2. (int): which shows the number of past steps used to predict the next step.

    :return:
        X(np.array): input sequence of shape (T, lookback, num_features) Each sample is a sequence of loopback timesteps
        y(np.array): Outputs the next step after each sequence.
    """

    X, y = [], []
    for i in range(len(data) - lookback):  # Take the consecutive rows as input
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def time_to_timedelta(t):
    """
    In this function we convert the Excel-exported time slot values into pandas Timedelta. This step is required because
    of the .xlsx file type which has some issues with formatting of columns with the Pandas library.This function is
    used to standardise the different formats(String,timestamp, and timedelta into pandas Timedelta).

    :param:
        t: (str, pd.Timestamp, pd.Timedelta)
    :return:
        (pd.Timedelta)
    """

    if isinstance(t, str):  # e.g. '00:05:00'
        return pd.to_timedelta(t)
    if isinstance(t, pd.Timedelta):  # e.g. pd.Timedelta('00:05:00')
        return t
    if isinstance(t, (pd.Timestamp, np.datetime64)): # e.g. pd.Timestamp('2021-01-01 00:05:00')
        t = pd.to_datetime(t).time()
    if hasattr(t, "hour") and hasattr(t, "minute") and hasattr(t, "second"):
        return pd.to_timedelta(f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}")
    return pd.to_timedelta(str(t))


def preprocess_data(file_path, scats_number, lookback=12, batch_size=32, train_start=None, train_end=None):
    """
    This function is used to preprocess the data. It loads the data from the Excel file, filters by SCATS_Number, and
    then creates the input-output sequence pairs for time series forecasting. The function returns the DataLoader object
    and the min/max values of the data.

    :param
        file_path: (str)
        scats_number: (int)
        lookback: (int)
        batch_size: (int)
        train_start: (str)
        train_end: (str)
    :return:
        train_loader: (DataLoader)
        min_val: (float)
        max_val: (float)
    """
    df = pd.read_excel(file_path)  # Load data sheet
    df = df[df['SCATS_Number'] == scats_number].copy()  # Filter by SCATS_Number

    # Combine duplicate data (SCATS_Number, Date, Weekday) by summing the volumes
    df = df.groupby(['SCATS_Number', 'Date', 'Weekday']).sum(numeric_only=True).reset_index()

    df_long = df.melt(  # Add datetime column
        id_vars=['Date', 'Weekday', 'SCATS_Number'],
        var_name='TimeSlot', value_name='Volume'
    )

    df_long['DateTime'] = pd.to_datetime(df_long['Date']) + df_long['TimeSlot'].apply(time_to_timedelta)  # Construct
    # timestamps = Date + TimeSlot
    df_long = df_long.sort_values('DateTime').reset_index(drop=True) # Chronological order check (Edge Case)

    # Temporal features
    seconds_in_day = 24*60*60
    df_long['Seconds'] = (
            df_long['DateTime'].dt.hour*3600 +
            df_long['DateTime'].dt.minute*60 +
            df_long['DateTime'].dt.second
    )
    # Encoding cylindrical Day data setting the sin and cos of the day to be between -1 and 1
    df_long['Time_sin'] = np.sin(2*np.pi*df_long['Seconds']/seconds_in_day)
    df_long['Time_cos'] = np.cos(2*np.pi*df_long['Seconds']/seconds_in_day)

    # Encoding cylindrical weekday data setting the sin and cos of the day to be between -1 and 1
    df_long['Weekday'] = pd.to_datetime(df_long['Date']).dt.weekday
    df_long['Weekday_sin'] = np.sin(2*np.pi*df_long['Weekday']/7)
    df_long['Weekday_cos'] = np.cos(2*np.pi*df_long['Weekday']/7)

    # Scaling the data
    traffic_data = df_long['Volume'].astype(float).values
    if train_start and train_end: # Scale only using training window min/max to avoid leakage
        train_mask = (df_long['DateTime'] >= pd.to_datetime(train_start)) & \
                     (df_long['DateTime'] <= pd.to_datetime(train_end))
        train_values = df_long.loc[train_mask, 'Volume'].astype(float).values
        min_val, max_val = train_values.min(), train_values.max()  # Scale the whole dataset min/max
    else:
        min_val, max_val = traffic_data.min(), traffic_data.max()

    df_long['Volume_norm'] = (traffic_data - min_val) / (max_val - min_val + 1e-8)  # Min-max scaling to [0, 1] range

    # Build the features
    features = df_long[['Volume_norm', 'Time_sin', 'Time_cos', 'Weekday_sin', 'Weekday_cos']].values

    # Sequence generation
    X, y = create_sequences(features, lookback)

    # Convert to Pytorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Target is only the traffic volume (column 0 of features)
    y_tensor = torch.tensor(y[:, 0], dtype=torch.float32).unsqueeze(-1)

    # Wrap into DataLoader for batching during testing (which is defined in the function statement)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, min_val, max_val