from torch.utils.data import DataLoader, ConcatDataset
from ml_models.lstm_model import train_model
from ml_models.lstm_data_handler import preprocess_data
import pandas as pd
import torch
import os


def combine_all_site_data(data_file_path, all_scats_numbers):
    """
    Combine training data from all SCATS sites into one dataset
    Returns combined DataLoader and global min/max values
    :param
    data_file_path: path to data file
    all_scats_numbers: list of SCATS site numbers
    :return: combined DataLoader, global min/max values, transform_dict
    """
    print("Combining data from all SCATS sites...")
    all_datasets = []
    all_min_vals = []
    all_max_vals = []

    # Collect datasets and normalization values from each site
    for scats_number in all_scats_numbers:
        print(f"  Processing SCATS site: {scats_number}")
        train_loader, min_val, max_val = preprocess_data(data_file_path, scats_number) # Get individual site data
        dataset = train_loader.dataset  # Extract the dataset from the DataLoader
        all_datasets.append(dataset)
        all_min_vals.append(min_val) # Collect min/max values for global normalization
        all_max_vals.append(max_val)

    # Calculate global min/max across all sites
    global_min = min(all_min_vals)
    global_max = max(all_max_vals)

    print(f"Global normalization range: min={global_min}, max={global_max}")
    print(f"Individual site min range: {min(all_min_vals)} - {max(all_min_vals)}")
    print(f"Individual site max range: {min(all_max_vals)} - {max(all_max_vals)}")

    combined_dataset = ConcatDataset(all_datasets)  # Combine all datasets to create new DataLoader
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
    print(f"Combined dataset size: {len(combined_dataset)} samples")
    print(f"Number of batches: {len(combined_loader)}")

    # Create transform_dict with global values
    transform_dict = {
        "min_val": global_min,
        "max_val": global_max,
        "max_tfv": global_max
    }
    return combined_loader, global_min, global_max, transform_dict


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Loading and preparing data...")
    data_file_path = "data/model_data.xlsx"
    df_scats = pd.read_excel(data_file_path)
    all_scats_numbers = df_scats['SCATS_Number'].unique().tolist()
    print(f"Found {len(all_scats_numbers)} SCATS sites: {all_scats_numbers}")
    final_hyperparameters = {
        'epochs': 100,           # with early stopping
        'learning_rate': 0.0005,
        'hidden_size': 150,
        'num_layers': 3,
        'optimizer': 'Adam'
    }
    os.makedirs("saved_models", exist_ok=True)
    # Clean up old LSTM models before training new combined model
    print("Cleaning up old LSTM models...")
    import glob
    lstm_files = glob.glob("saved_models/lstm*.pth")
    lstm_files = [f for f in lstm_files if not any(model_type in f.lower() for model_type in ['gru', 'tcn'])]
    for file_path in lstm_files:
        try:
            os.remove(file_path)
            print(f"  Deleted: {file_path}")
        except OSError as e:
            print(f"  Error deleting {file_path}: {e}")
    if lstm_files:
        print(f"Deleted {len(lstm_files)} old LSTM model files")
        print("Preserved GRU and TCN models")
    else:
        print("No old LSTM models found to delete")
    combined_train_loader, global_min, global_max, transform_dict = combine_all_site_data(
        data_file_path, all_scats_numbers
    )
    print(f"\nTraining single LSTM model on combined data from {len(all_scats_numbers)} sites...")
    print(f"Using global normalization: min={global_min}, max={global_max}")
    # Train one model on all combined data
    final_flowrate_model, final_accuracy = train_model(
        combined_train_loader,
        global_min,
        global_max,
        final_hyperparameters,
        transform_dict=transform_dict
    )
    # Save the single combined model
    model_path = "saved_models/lstm.pth"
    torch.save(final_flowrate_model._model, model_path)
    print(f"Saved combined LSTM model at {model_path}")
    print(f"\nCombined Model Training Complete!")
    print(f"Final model accuracy (MAE): {final_accuracy:.4f}")