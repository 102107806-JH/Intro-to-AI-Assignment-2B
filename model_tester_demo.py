from prototype_model_tester.test_data_store import TestDataStore
from datetime import datetime, timedelta
from prototype_model_tester.flowrate_prediction_tester import FlowratePredictionTester
import matplotlib.pyplot as plt
import numpy as np
import math
if __name__ == "__main__":
    prediction_depth = 1

    test_data_store = TestDataStore(database_file_path="data/data_base.xlsx")
    flowrate_prediction_tester = FlowratePredictionTester()

    scats_site = 970
    time_of_prediction = datetime(year=2025, month=8, day=1, hour=0, minute=0)
    final_prediction_exclusive = datetime(year=2025, month=8, day=8, hour=0, minute=0)
    time_dif = final_prediction_exclusive - time_of_prediction
    time_dif_total_seconds = time_dif.days * 24 * 60 * 60 + time_dif.seconds
    number_of_predictions = + math.floor((time_dif_total_seconds / 60) / 15)
    yhat_list_gru = []
    yhat_list_tcn = []
    y_list = []
    time_of_prediction_list = []
    for i in range(number_of_predictions):
        sequence = test_data_store.query_sequence(query_time=time_of_prediction - timedelta(minutes=15*prediction_depth),scats_site=scats_site, sequence_length=12)

        yhat_gru = flowrate_prediction_tester.test(sequence, scats_site=scats_site, mode="GRU", prediction_depth=prediction_depth)
        yhat_tcn = flowrate_prediction_tester.test(sequence, scats_site=scats_site, mode="TCN", prediction_depth=prediction_depth)
        y = test_data_store.query(query_time=time_of_prediction, scats_site=scats_site)

        yhat_list_gru.append(yhat_gru)
        yhat_list_tcn.append(yhat_tcn)
        y_list.append(y)
        time_of_prediction_list.append(time_of_prediction)
        time_of_prediction = time_of_prediction + timedelta(minutes=15)

    average_dif_gru = np.average(np.abs(np.array(y_list) - np.array(yhat_list_gru)))
    average_dif_tcn = np.average(np.abs(np.array(y_list) - np.array(yhat_list_tcn)))
    print(f"AVERAGE ABS DIFFERENCE FROM ACTUAL:\nGRU:{average_dif_gru}\nTCN:{average_dif_tcn}")
    plt.plot(y_list,'g')
    plt.plot(yhat_list_gru, 'r')
    plt.plot(yhat_list_tcn, 'b')
    plt.show()
