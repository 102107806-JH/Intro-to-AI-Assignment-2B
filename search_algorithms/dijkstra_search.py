from textbook_abstractions.node import Node
from data_structures.queues.priority_que import PriorityQueue
from misc.mock_model import MockModel

def dijkstras_serach(problem):
    node = Node(state=problem.initial_state)  # Put the initial state inside a node
    frontier = PriorityQueue(key_lambda=lambda node:(node.total_cost, node.state, node.order_pushed_into_collection))  # Create frontier and establish sorting order #
    frontier.push(node)  # Push the node onto the frontier #
    reached = {problem.initial_state: node}  # Dictionary that stores the reached state states #
    solution_nodes = []  # List to store all the different solution nodes #
    while frontier.is_empty() == False:  # Keep going until the frontier is empty #
        node = frontier.pop()  # Pop the top off the frontier #

        if problem.is_goal(node.state):  # The goal has been found #
            solution_nodes.append(node)  # Append the solution node to the solution node list #
            continue  # Because we want to be able to find multiple roots we keep looking for more roots we use the continue though because we don't wish to expand the solution node #

        for child in expand(problem, node):  # Expand the nodes children
            state = child.state  # Get the child node state #
            if state not in reached or child.path_cost < reached[state].path_cost:  # If the state is not in reached or the path is better push the child and add it to reached
                reached[state] = child
                frontier.push(child)
    return solution_nodes

def expand(problem, node):
    state = node.state
    children = []

    for action in problem.actions(state):
        new_state = problem.result(state, action)
        path_cost = node.path_cost + problem.action_cost(state, action, new_state)
        children.append(Node(state=new_state, parent=node, action=action, path_cost=path_cost, total_cost=path_cost))

    return children