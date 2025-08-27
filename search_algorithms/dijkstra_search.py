from textbook_abstractions.node import Node
from data_structures.queues.priority_que import PriorityQueue

def dijkstras_serach(problem):
    node = Node(state=problem.initial_state)
    frontier = PriorityQueue(key_lambda=lambda node:(node.total_cost, node.state, node.order_pushed_into_collection))
    reached = {problem.initial_state: node}  # Dictionary that stores the reached state states #

    while frontier.is_empty() == False:
        node = frontier.pop()

        if problem.is_goal(node.state):
            return node

        for child in expand(problem, node):
            s = child.state

def expand(problem, node)