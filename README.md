# Intro-to-AI-Assignment-2B Machine learning and software integration Pipeline
## **IMPORTANT READ BEFORE USING ANY PART OF THIS PROJECT**
prior to using the program it is crucial that you wait for the webpage to load in-between actions. On the site the tab will say “Updating...” whilst this is happening do not perform any other interactions with the site. If you use the play button, you must pause press the pause button and let the tab update again before entering different parameters (discussed more in the timeline section of the readme). Simply after each action always look to the tab and if it says “Updating...” wait for it to update if you do perform an action during updating it can have unforeseen consequences to fix this close the webpage and halt execution of the app and then simply restart it. Running the training, scripts will change the models saved after they have executed so if this needs to be done do it after testing the other components of the program or alternatively clone the repository again to get access to the original models. Training scripts may take a long time when run on the CPU as the have been designed to run on the GPU.
## Features
- **Multiple Deep-Learning Model Types**: LSTM, GRU, and TCN. 
- **Comprehensive Evaluation**:
- **Automated Reporting**: CSV exports, comprehensive visualizations, and per-SCAT Site metrics charts.
- **Modular Architecture**: Clean separation of data processing, training, evaluation, and GUI

## GUI Interface
### About the GUI
The GUI is a web-based interface that allows users to interact with each model. It was built using the Dash Python framework.

### First Loading the GUI
To use the GUI effectively, please allow time for all page elements to load. On the left side of the screen, there are adjustable parameters that update the GUI periodically.

### Interactive Map Controls
Upon loading the interactive map, you can zoom in and out, pan, and reset the map to the full extent of the data using either your mouse and mouse wheel or the navigation buttons in the top right corner. The red border shows the city limits of the Boroondara area. You can also hover over any of the SCATS sites, indicated by a circle on the map, to see more information about the site. This includes the site's number, the current hour, the volume of traffic during that hour, the delta change in cars from the previous hour, and the percentage of change.

The sites are also colour-coded: yellow indicates an increase in traffic volume, blue indicates a decrease in traffic volume, white indicates the peak traffic volume for a given day, and grey indicates the lowest traffic volume for a given day.

### Origin SCATS Site Parameters
The Origin SCATS Site is the starting point of your journey. You can select a new Origin SCATS Site by clicking the drop-down menu and scrolling with either your mouse wheel or the scroll bar. You can also type the SCATS site number in the text box to quickly filter the list.

### Destination SCATS Site Parameters
Below the Origin is the Destination SCATS Site. As mentioned previously, you can select a new Destination SCATS Site by clicking the drop-down menu and scrolling. You can also type the SCATS site number in the text box to quickly filter the list. Below the Destination is the Model Type. You can select a new Model Type by clicking the drop-down menu. You can also type the Model Type to quickly filter the list of available models.

### Sequence Length
The Sequence Length is the number of time steps used in the model. You cannot change the sequence length, as this is outside the scope of the project. However, it is displayed for your convenience.

### K-Value
Below the sequence length is the K-Value, which is the number of paths that will be generated for the selected model. Each path will be displayed in a different colour to allow for easy visual comparison. Please note that paths may overlap. You can filter them on the right side of the screen by clicking on a desired path to hide the others.

### Finding the Paths
To calculate and find the path, please press the Find Path button below the K-Value input field. This will cause the page to update. The progress of the page update is shown at the top of the page, where the tab is temporarily renamed from "Dash" to "Updating." During this time, please do not change any parameters, as this will cause the page to update again before displaying the previous results. When clicking the Find Path button, also ensure that the "hour of the day" timeline is paused, not playing, as this will cause the page to continuously check for updates and will not load the results.

### Map Visualisation Sidebar
Once you click the Find Path button and the page finishes updating, a new sidebar will appear on the right side of the map. At the top of this sidebar, the current traffic volume status for the selected Origin SCATS site is displayed.

Below that is the most optimal path, shown in purple. Subsequent paths are shown in random colours for easy visual comparison. To view only one path when multiple are generated, or to compare different paths, you can click on the path's name in the legend to show or hide it. The estimated time of arrival is shown to the right of the path's name in an hourly format (e.g., 0.25 hours is equal to 25 minutes, and 0.5 hours is equal to 30 minutes).

### Timeline
Below the Find Path button is the "hour of the day" timeline. As part of our custom research initiative, we have implemented a feature that allows you to see how busy a SCATS site is at a given hour. You can select a new hour by clicking and dragging the blue timeline. You can also change the date by clicking on the date drop-down menu, which also supports typing to filter the dates. Additionally, you can click the Play button to automatically play back the entire day hour by hour. To update the GUI with new parameters, you must press Pause and wait for the GUI to finish updating. The current date and time are displayed to the right of the Pause button.

### ABS Interactive Visualisation
Below the Pause button is an interactive visualisation that plots the ABS (Average Absolute Difference from Actual). The data shows the calculated difference for each model for the selected Origin SCATS site, where a lower value indicates better accuracy. For example, a score of 10.04 means that, on average, the model's predictions for traffic flow at that location were off by approximately 10.04 TFV (traffic flow volume) units. This value is a measure of the model's prediction error.

The X-axis of the visualisation shows the sequential prediction step made by the models. The Y-axis shows the ABS difference between the model's prediction and the actual traffic flow. The colour of the line indicates which model made the prediction. You can click and drag on the visualisation to zoom, pan, autoscale, reset the axes, and download it as a PNG using your mouse or the navigation buttons in the top right corner. Additionally, hovering over any data point will provide more details about the model's absolute error at that specific timestep.

## How to Run the Program
### 1. Create a virtual environment (Optional)
```bash
# This step is for machines that do not use a Windows 10/11 operating system: Do not run this script on a Windows 10/11 machine.

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies 
# [IF THIS STEP DOES NOT INSTALL ALL OF THE REQUIRED REQUIREMENTS, THEN USE PIP INSTALL FOR EACH OF THE REQUIREMENTS AT THE BOTTOM OF THE README] 
pip install -r requirements.txt
```

### 2. Test CUDA GPU availability (Optional)
````
# If you are attempting to train/re-train the models then it is highly recommended that you have a CUDA GPU available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
````

### 3. Run the LSTM Individual Model Training (Optional)
````
# The LSTM Model has already been trained and the model has been saved, but this can re-train the model if needed. 
# Train the LSTM Model for each SCATS site
python LSTM_Train_Multible_Model.py
````

### 4. Run the LSTM Combined Model Training (Optional)
````
# You do not need to run this if you are not training the models or have not completed step 3.
# Combines the previously trained LSTM models and re-trains them into one model, that's universal for every SCATS site
python LSTM_Train_Combined_Model.py
````
### 5. Hyper-Parameter Tuning Script
````
# Runs a script that tests random hyper-parameter combinations and returns the best combination at the end of the alloted time period. The finishing time and the different hyper-parameter values to test can be set inside the script.
python hyper_parameter_tuning_script.py
````
### 6. Run the GRU and TCN Training (Optional)
````
# The GRU and TCN models have already been trained and the model has been saved, but this can re-train the model if needed.
# Used to train the GRU and the TCN models. The model that is selected for training and the hyperparameters used can be changed inside the script.
python GRU_TCN_models.py
````
### 7. Test Model Performance
````
# Displays a graph that plots the ABS difference for a prefered SCAT site, which can be adjusted in the model_tester_demo.py file.
python model_tester_demo.py
````
### 8. Test All Models On Deployment Data (Optional)
````
# Tests all models on deployment data and saves the results into a dictionary inside the testing/deployment_data_test_results directory
python model_deployment_data_tester.py
````
### 9. Test All Models On Deployment Data
````
# Loads the data created by 'model_deployment_data_tester.py' and performs model evaluation that was used in the report.
python model_evaluation_script.py
````
### 10. Path finding demonstration (Optional)
````
# Within the python file you can enter different parameters for the pathfinding function. The path finding function will then find all the solution nodes and print them out in the console.
python path_finding_demo.py
````
**PLEASE NOTE ALL PATHFINDING AND PREDICTION ALGORITHMS USE THE DATA IN 'DATA_BASE.XLSX'. THE DATA PROCESSING SCRIPTS IN 9 AND 10 WERE USED TO CREATE THIS. THE ONLY OTHER PROCESSING DONE WAS ADDING COLUMN NAMES OUTSIDE OF PYTHON. ANY FILES CREATED WILL NOT HAVE THE NAME 'TEST' IN THEIR TO AVOID CORRUPTING DATA. FURTHERMORE, THESE SCRIPTS MAY TAKE A LONG TIME TO FULLY EXECUTE**
### 11. Mock database creator
````
# Extracted the traffic flow volume data from the months of 08/2025 and 09/2025. The files which it extracts data from are not present as there were too many of them. This file has been included for the sake of completeness.
python mock_database_creator_script.py
````
### 12. Mock database cleaner (Optional)
````
# The the traffic flow volume data from the months of 08/2025 and 09/2025 had some corrupted running this script removes all -1's and 0 rows. 
python mock_database_cleaner_script.py
````

### 13. Run the GUI
````
# Run the interactive Dash GUI
python app.py
````


## Project Structure
````
|-- Project Main Folder
|-- .gitignore
|-- app.py
|-- GRU_TCN_models.py
|-- hyper_parameter_tuning_script.py
|-- LSTM_Train_Combined_Model.py
|-- LSTM_Train_Multible_Models.py
|-- mock_database_cleaner_script.py
|-- mock_database_creator_script.py
|-- model_deployment_data_tester.py
|-- model_evaluation_script.py
|-- model_tester_demo.py
|-- path_finding_demo.py
|-- README.md
|-- requirements.txt
|-- data
|   |-- traffic_signal_volume_cur
|   |   |-- VSDATA_20250701.csv
|   |   |-- VSDATA_20250831.csv
|   |-- data_base.xlsx
|   |-- graph_init_data.xlsx
|   |-- model_data.xlsx
|   |-- uncleaned_data_base.xlsx
|-- data_structures
|   |-- graph_classes
|   |   |-- adjacency_list_graph.py
|   |   |-- destination_distance_pair.py
|   |   |-- vertex.py
|   |   |-- linked_list
|   |   |   |-- list_node.py
|   |   |   |-- singly_linked_list.py
|   |-- queues
|       |-- priority_que.py
|-- file_handling
|   |-- graph_vertex_edge_init.py
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
|-- jh_ml_models
|   |-- model_code
|   |   |-- gru_model.py
|   |   |-- model_collection.py
|   |   |-- tcn_model.py
|   |-- model_deployment_abstractions
|   |   |-- current_deployment_data_store.py
|   |   |-- flowrate_predictor.py
|   |   |-- deployment_data_testing
|   |   |   |-- deployment_data_model_tester.py
|   |   |   |-- flowrate_prediction_tester.py
|   |   |   |-- test_deployment_data_store.py
|   |   |   |-- multi_scat_site_tester.py
|   |-- model_fitting
|   |   |-- data_loader.py
|   |   |-- hyper_parameter_tuner.py
|   |   |-- model_fitter.py
|   |   |-- hyper_parameter_tuning_results
|   |   |   |--GRU_01.txt
|   |   |   |--GRU_02.txt
|   |   |   |--TCN02.txt
|   |   |   |--TCN_01.txt 
|-- misc
|   |-- mock_data_base_creator.py
|   |-- mock_data_base_cleaner.py
|-- ml_models
|   |-- lstm_data_handler.py
|   |-- lstm_model.py
|-- path_finding
|   |-- path_finder.py
|-- saved_models
|   |-- gru.pth
|   |-- lstm.pth
|   |-- tcn.pth
|-- textbook_abstractions
|   |-- node.py
|   |-- problem.py
````


## Requirements
# Please use PIP install for each of the following libraries if they are not already installed
- colorama
- ConfigParser
- cryptography
- dash
- dash-core-components
- docutils
- filelock
- fonttools
- geopandas
- HTMLParser
- ipython
- ipywidgets
- matplotlib
- numpy
- openpyxl
- pandas
- Pillow
- plotly
- protobuf
- pyOpenSSL
- redis
- Sphinx
- thread
- torch
- torchaudio
- torchvision
- tqdm
- urllib3_secure_extra
- xlsxwriter

See `requirements.txt` for specific versions, this is optional.