from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from textbook_abstractions.problem import Problem
from textbook_abstractions.node import Node
from search_algorithms.dijkstra_search import dijkstras_serach
from misc.mock_model import MockModel
if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()  # Extract all the data from the graph init file and store in graph #
    mock_flowrate_prediction_model = MockModel()  # init prediction model (this is a mock model for the time being but will be replaced with actual ones)
    test_problem = Problem(graph, initial_state=3662, goal_state=2000,flowrate_prediction_model=mock_flowrate_prediction_model)  # Create the problem object which as an instance of one of the text-book abstractions #
    solution_nodes = dijkstras_serach(test_problem, k_val=100)  # Return all solution nodes #
    print("Program End")