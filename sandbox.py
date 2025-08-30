from ml_models.jh_data_loader import TrafficFlowDataSet

if __name__ == "__main__":
    test = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=3)