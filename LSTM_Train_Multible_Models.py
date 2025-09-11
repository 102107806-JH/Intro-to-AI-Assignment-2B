from ml_models.lstm_model import train_model
from ml_models.lstm_data_handler import preprocess_data
import pandas as pd
import torch
import os

"""
Train a LSTM model for each SCATS site, this ensures that each SCAT site is accessible to all SCAT sites. 
:param
    data_file_path: path to data file
:return
    final_flowrate_model: trained LSTM model
    best_models: dictionary of best models for each SCATS site
"""

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
    best_models = {}

    for scats_number in all_scats_numbers:
        print(f"\nTraining model for SCATS site: {scats_number}")
        train_loader, min_val, max_val = preprocess_data(data_file_path, scats_number)
        transform_dict = {"min_val": min_val, "max_val": max_val, "max_tfv": max_val}
        final_flowrate_model, final_accuracy = train_model(
            train_loader,
            min_val,
            max_val,
            final_hyperparameters,
            transform_dict=transform_dict
        )

        # Save per-SCATS model
        model_path = f"saved_models/lstm_{scats_number}.pth"
        torch.save(final_flowrate_model._model, model_path)
        print(f"Saved LSTM model for SCATS {scats_number} at {model_path}")
        best_models[scats_number] = {
            'model': final_flowrate_model,
            'accuracy': final_accuracy,
            'hyperparameters': final_hyperparameters
        }
    print(f"\n Individual Model Training Complete for All {len(all_scats_numbers)} Sites")
    # Print summary of all models
    print("\nModel Performance Summary:")
    print("-" * 50)
    for scats_number, model_info in best_models.items():
        print(f"SCATS {scats_number}: MAE = {model_info['accuracy']:.4f}")