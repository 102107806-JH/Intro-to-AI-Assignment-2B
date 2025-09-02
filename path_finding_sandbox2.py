from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from jh_ml_models.flowrate_predictor import FlowratePredictor
import time
from datetime import date, datetime

if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    flowrate_predictor = FlowratePredictor()
    current_time = datetime.now()
    path_finder = PathFinder(graph=graph, flowrate_predictor=flowrate_predictor)
    path_finder.find_paths(initial_state=3662, goal_state=2000, current_time=current_time, k_val=100)
    print("Program End")