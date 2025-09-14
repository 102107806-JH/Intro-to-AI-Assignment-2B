from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import datetime

def print_solution(node):
    scats_site_list = []
    print(f"Hours to get to goal scats site: {node.path_cost}")
    while node is not None:
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
    current_time = current_time.replace(day=15)

    current_time = current_time.replace(hour=18)
    path_finder = PathFinder(graph=graph)

    k_val = 10 # The number of solutions that you want to find
    initial_state = 2000  # Initial state from the below list
    goal_state = 4821 # Goal state from the below list
    mode = "TCN" # LSTM, GRU or TCN
    # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120,
    # 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804, 3812, 4030,
    # 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263,
    # 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821,
    solution_nodes = path_finder.find_paths(initial_state=initial_state, goal_state=goal_state, initial_time=current_time, sequence_length=12, k_val=k_val, mode=mode)
    for i, node in enumerate(solution_nodes):
        print("__________________________________________________")
        print(f"Solution Number {i + 1}")
        print_solution(node)

