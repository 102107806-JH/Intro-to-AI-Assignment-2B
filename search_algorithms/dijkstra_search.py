from textbook_abstractions.node import Node
from data_structures.queues.priority_que import PriorityQueue
from misc.mock_model import MockModel

def dijkstras_serach(problem):
    node = Node(state=problem.initial_state)
    frontier = PriorityQueue(key_lambda=lambda node:(node.total_cost, node.state, node.order_pushed_into_collection))
    frontier.push(node)
    reached = {problem.initial_state: node}  # Dictionary that stores the reached state states #
    while frontier.is_empty() == False:
        node = frontier.pop()

        if problem.is_goal(node.state):
            return node

        for child in expand(problem, node):
            state = child.state
            if state not in reached or child.path_cost < reached[state].path_cost:
                reached[state] = child
                frontier.push(child)
    return None

def expand(problem, node):
    state = node.state
    children = []
    mock_model = MockModel()

    for action in problem.actions(state):
        new_state = problem.result(state, action)
        path_cost = node.path_cost + problem.action_cost(state, action, new_state) + mock_model.make_prediction()
        children.append(Node(state=new_state, parent=node, action=action, path_cost=path_cost, total_cost=path_cost))

    return children