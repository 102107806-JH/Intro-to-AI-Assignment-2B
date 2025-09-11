from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import datetime

def print_solution(node):
    scats_site_list = []
    print(f"Hours to get to goal scats site: {node.path_cost}")
    while node.parent is not None:
        scats_site_list.append(node.state)
        node = node.parent

    print("Scat site path: ", end="")
    for scats_site in reversed(scats_site_list):
        if scats_site != scats_site_list[0]:
            print(f"{scats_site}->", end="")
        else:
            print(f"{scats_site}")


if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    current_time = datetime.now()
    current_time = current_time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
    path_finder = PathFinder(graph=graph)
    solution_nodes = path_finder.find_paths(initial_state=2000, goal_state=970, initial_time=current_time, sequence_length=12, k_val=3, mode="LSTM")
    for i, node in enumerate(solution_nodes):
        print("__________________________________________________")
        print(f"Solution Number {i + 1}")

        print_solution(node)

