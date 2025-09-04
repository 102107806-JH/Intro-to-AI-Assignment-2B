from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import date, datetime, timedelta
from jh_ml_models.flowrate_predictor import FlowratePredictor
from data_handling.current_data import CurrentData
import matplotlib.pyplot as plt
if __name__ == "__main__":
    current_time = datetime(2025,8,5,12,0,0)
    current_time = current_time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month

    flowrate_predictor = FlowratePredictor(initial_time=current_time,
                                           sequence_length=12,
                                           database_file_path="data/data_base.xlsx",
                                           mode="TCN")

    current_data = CurrentData(database_file_path="data/data_base.xlsx", initial_time=current_time + timedelta(hours=48))

    scats_site = 3002

    flow_rate_list_predicted = []
    flow_rate_list_actual = []
    delta_t = 0
    delta_t2 = 0
    for _ in range(96):
        flow_rate_list_predicted.append(flowrate_predictor.get_data(time_since_initial_time=delta_t, scats_site=scats_site))
        flow_rate_list_actual.append(current_data.query(query_time=current_time + timedelta(minutes=delta_t2), scats_site=scats_site))
        delta_t += 0.25
        delta_t2 += 15
    plt.plot(flow_rate_list_predicted, 'r', flow_rate_list_actual, 'g')
    plt.show()
    print("Program End")