from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from jh_ml_models.model_deployment_abstractions.deployment_data_testing.deployment_data_model_tester import DeploymentDataModelTester

if __name__ == "__main__":
    start_datetime = datetime(year=2025, month=8, day=1, hour=0, minute=0)
    end_datetime = datetime(year=2025, month=8, day=2, hour=0,minute=0)
    model_tester = DeploymentDataModelTester(database_file_path="data/data_base.xlsx")
    results = model_tester.test_models(
        scats_site=970,
        prediction_depth=1,
        sequence_length=12,
        start_datetime=start_datetime,
        end_datetime=end_datetime)

    average_dif_gru = np.average(np.abs(np.array(results["Targets"]) - np.array(results["GRU"])))
    average_dif_tcn = np.average(np.abs(np.array(results["Targets"]) - np.array(results["TCN"])))
    average_dif_lstm = np.average(np.abs(np.array(results["Targets"]) - np.array(results["LSTM"])))
    print(f"AVERAGE ABS DIFFERENCE FROM ACTUAL:\nGRU:{average_dif_gru}\nTCN:{average_dif_tcn}\nLSTM:{average_dif_lstm}")
    plt.plot(results["Targets"],'g')
    plt.plot(results["GRU"], 'r')
    plt.plot(results["TCN"], 'b')
    plt.plot(results["LSTM"],'m')
    plt.show()