from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import date, datetime
from jh_ml_models.flowrate_predictor import FlowratePredictor
if __name__ == "__main__":
    current_time = datetime.now()
    current_time = current_time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month

    flowrate_predictor = FlowratePredictor(initial_time=current_time,
                                           sequence_length=12,
                                           database_file_path="data/data_base.xlsx",
                                           mode="GRU")

    flowrate_predictor.get_data(time_since_initial_time=0.3, scats_site=2000)

    print("Program End")