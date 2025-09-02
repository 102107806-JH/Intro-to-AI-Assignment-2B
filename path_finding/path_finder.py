from fontTools.merge.util import current_time

from textbook_abstractions.node import Node
from data_structures.queues.priority_que import PriorityQueue
from helper_functions.cycle_checker import cycle_checker
from datetime import date, datetime

from textbook_abstractions.problem import Problem

class PathFinder():
    def __init__(self, graph, flowrate_predictor):
        self._graph = graph
        self._flowrate_predictor = flowrate_predictor

    def find_paths(self, initial_state, goal_state, current_time=datetime.now(), k_val=1):
        self._flowrate_predictor.set_initial_time(current_time)

        problem = Problem(graph=self._graph,
                          initial_state=initial_state,
                          goal_state=goal_state,
                          flowrate_predictor=self._flowrate_predictor)

        soloution_nodes = self._dijkstras_serach(problem=problem, k_val=k_val)

    def _dijkstras_serach(self, problem, k_val):
        """
        This is a modified version of dijkstras search. The top k paths needed to
        be found. This version does not maintain a dictionary of visited states.
        This is because if it did it would prevent it would prevent potential paths
        in the top k paths from being explored. It uses a cycle checker to check
        to the root node of the tree which prevents cycling an ensures that the
        algorithm does not get stuck in an infinite loop. The algorithm halts when
        either the frontier is empty or k solutions have been found.
        """
        node = Node(state=problem.initial_state)  # Put the initial state inside a node
        frontier = PriorityQueue(key_lambda=lambda node: (node.total_cost, node.state,node.order_pushed_into_collection))  # Create frontier and establish sorting order #
        frontier.push(node)  # Push the node onto the frontier #
        solution_nodes = []  # List to store all the different solution nodes #

        while frontier.is_empty() == False:  # Keep going until the frontier is empty #
            node = frontier.pop()  # Pop the top off the frontier #

            if problem.is_goal(node.state):  # The goal has been found #
                solution_nodes.append(node)  # Append the solution node to the solution node list #
                if len(solution_nodes) == k_val:  # The desired number of solutions has been found so break out of the while loop #
                    break
                continue  # Because we want to be able to find multiple paths we keep looking for more paths we use the continue though because we don't wish to expand the solution node #

            for child in self._expand(problem, node):  # Expand the nodes children

                if cycle_checker(child):  # No cycles are allowed. If detected continue to ensure child node is not pushed onto the frontier #
                    continue

                frontier.push(child)  # Push child onto the frontier

        return solution_nodes

    def _expand(self, problem, node):  # Expand the node #
        state = node.state  # State of node #
        children = []  # The list to store all the children nodes #
        data = None
        for action in problem.actions(state):  # Going through all the actions available from the current state #
            new_state = problem.result(state,action)  # The resulting state from the action #
            elapsed_time = node.path_cost  # The time elapsed to get to the parent node #
            path_cost = elapsed_time + problem.action_cost(state, action, new_state, elapsed_time)  # The path cost to the new state #
            children.append(Node(state=new_state, parent=node, action=action, path_cost=path_cost, total_cost=path_cost))  # Creating a new child #

        return children  # Return all the children #

