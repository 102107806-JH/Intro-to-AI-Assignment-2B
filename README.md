# Intro-to-AI-Assignment-2B Machine learning and software integration Pipeline

## Features
- **Multible Deep-Learning Model Types**: LSTM, GRU, and TCN. 
- **Comprehensive Evaluation**:
- **Automated Reporting**: CSV exports, comprehensive visualizations, and per-SCAT Site metrics charts.
- - **Modular Architecture**: Clean separation of data processing, training, evaluation, and GUI

## Quickstart Guide

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test CUDA GPU availability
````
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
````

### 3. Run the LSTM Individual Model Training (Optional)
````
# Train the LSTM Model for each SCATS site
python train_individual_models.py
````

### 4. Run the LSTM Combined Model Training (Optional)
````
# Combines the previously trained LSTM models and re-trains them into one model, that's universal for every SCATS site
python evaluate_individual_models.py
````

### ?. Test all Models Performance
````
# Displays a graph that plots the ABS difference for a prefered SCAT site, which can be adjusted in the model_tester_demo.py file.
python model_tester_demo.py
````

### ?. Run the GUI
````
# Run the interactive Dash GUI
python app.py
````


## Project Structure
````
|-- Project Main Folder
|-- .gitignore
|-- app.py
|-- database_creator.py
|-- jh_training_script.py
|-- LSTM_Train_Combined_Model.py
|-- LSTM_Train_Multible_Models.py
|-- model_tester_demo.py
|-- path_finding_demo.py
|-- README.md
|-- requirements.txt
|-- data
|   |-- data_base.xlsx
|   |-- graph_init_data.xlsx
|   |-- model_data.xlsx
|   |-- ~$data_base.xlsx
|   |-- ~$model_data.xlsx
|-- data_structures
|   |-- graph_classes
|   |   |-- adjacency_list_graph.py
|   |   |-- destination_distance_pair.py
|   |   |-- vertex.py
|   |   |-- linked_list
|   |   |   |-- list_node.py
|   |   |   |-- singly_linked_list.py
|   |   |   |--
|   |   |       |-- list_node.cpython
|   |   |       |-- singly_linked_list.cpython
|   |   |--
|   |       |-- adjacency_list_graph.cpython
|   |       |-- destination_distance_pair.cpython
|   |       |-- vertex.cpython
|   |-- queues
|       |-- priority_que.py
|       |--
|           |-- priority_que.cpython
|-- file_handling
|   |-- graph_vertex_edge_init.py
|   |--
|       |-- graph_vertex_edge_init.cpython
|-- GUI
|   |-- graph_init_data.xlsx
|   |-- model_data.xlsx
|   |-- vic_lga.dbf
|   |-- vic_lga.prj
|   |-- vic_lga.shp
|   |-- vic_lga.shx
|   |-- vic_lga_locality.dbf
|   |-- cache
|       |-- ad0096fdfc4550ffa30b974fefa9a6a657bc949a.json
|       |-- c36450edb58e0e191825a6e452dde8b8f7e5c436.json
|       |-- d1fb2cd5aad1c6e024dfaa17b11d027026c73e2f.json
|-- helper_functions
|   |-- cycle_checker.py
|   |-- haversine.py
|   |--
|       |-- cycle_checker.cpython
|       |-- haversine.cpython
|-- jh_ml_models
|   |-- model_code
|   |   |-- gru_model.py
|   |   |-- model_collection.py
|   |   |-- tcn_model.py
|   |   |--
|   |       |-- gru_model.cpython
|   |       |-- lstm_data_handler.cpython
|   |       |-- model_collection.cpython
|   |       |-- tcn_model.cpython
|   |-- model_deployment_abstractions
|   |   |-- current_deployment_data_store.py
|   |   |-- flowrate_predictor.py
|   |   |-- deployment_data_testing
|   |   |   |-- deployment_data_model_tester.py
|   |   |   |-- flowrate_prediction_tester.py
|   |   |   |-- test_deployment_data_store.py
|   |   |   |--
|   |   |       |-- deployment_data_model_tester.cpython
|   |   |       |-- flowrate_prediction_tester.cpython
|   |   |       |-- test_deployment_data_store.cpython
|   |   |--
|   |       |-- current_deployment_data_store.cpython
|   |       |-- flowrate_predictor.cpython
|   |-- model_fitting
|   |   |-- data_loader.py
|   |   |-- model_fitter.py
|   |   |--
|   |--
|       |-- flowrate_predictor.cpython
|       |-- gru_model.cpython
|       |-- tcn_model.cpython
|-- misc
|   |-- mock_data_base_creator.py
|   |--
|       |-- mock_model.cpython
|-- ml_models
|   |-- lstm_data_handler.py
|   |-- lstm_model.py
|   |--
|       |-- lstm_data_handler.cpython
|       |-- lstm_model.cpython
|-- path_finding
|   |-- path_finder.py
|   |--
|       |-- path_finder.cpython
|-- saved_models
|   |-- gru.pth
|   |-- lstm.pth
|   |-- tcn.pth
|-- textbook_abstractions
|-- node.py
|-- problem.py
````


## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- os
- datetime
- math
- torch
- tqdm
- re
- openpyxl
- copy
- xlsxwriter
- geopandas
- plotly.graph_objects
- dash
- dash-core-components
- scikit-learn
- glob
- torch.utils.data.DataLoader
- ConcatDataset

See `requirements.txt` for specific versions.