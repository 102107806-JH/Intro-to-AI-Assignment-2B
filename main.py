from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from textbook_abstractions.problem import Problem
from textbook_abstractions.node import Node
from search_algorithms.dijkstra_search import dijkstras_serach
if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    test_problem = Problem(graph, initial_state=2200, goal_state=970)
    solution_node = dijkstras_serach(test_problem)