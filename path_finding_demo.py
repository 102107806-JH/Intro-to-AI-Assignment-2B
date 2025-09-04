from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import datetime

if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    current_time = datetime.now()
    current_time = current_time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
    path_finder = PathFinder(graph=graph)
    solution_nodes = path_finder.find_paths(initial_state=3662, goal_state=2000, initial_time=current_time, sequence_length=12, k_val=100, mode="GRU")
    print("Program End")