# Intro-to-AI-Assignment-2B Machine learning and software integration Pipeline

## Features
- **Multible Deep-Learning Model Types**: LSTM, GRU, and TCN. 
- **Comprehensive Evaluation**:
- **Automated Reporting**: CSV exports, comprehensive visualizations, and per-class metrics charts.
- - **Modular Architecture**: Clean separation of data processing, training, evaluation, and serving

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
python model_tester_demo.py
````


## Project Structure




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