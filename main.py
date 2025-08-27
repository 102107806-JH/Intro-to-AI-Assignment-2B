from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from textbook_abstractions.problem import Problem
from textbook_abstractions.node import Node
from search_algorithms.dijkstra_search import dijkstras_serach
from misc.mock_model import MockModel
if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    mock_flowrate_prediction_model = MockModel()
    test_problem = Problem(graph, initial_state=3662, goal_state=4040,flowrate_prediction_model=mock_flowrate_prediction_model)
    solution_nodes = dijkstras_serach(test_problem)
    print("end")