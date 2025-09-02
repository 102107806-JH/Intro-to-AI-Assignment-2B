import math
import pandas as pd
import numpy as np
from data_structures.graph_classes.linked_list.list_node import ListNode
from data_structures.graph_classes.linked_list.singly_linked_list import SinglyLinkedList
from data_structures.graph_classes.adjacency_list_graph import AdjacencyListGraph
from data_structures.graph_classes.vertex import Vertex
from data_structures.graph_classes.destination_distance_pair import DestinationDistancePair
from helper_functions.haversine import haversine

class Problem:
    def __init__(self, graph, initial_state, goal_state, flowrate_prediction_model):
        self._graph = graph
        self._initial_state = initial_state
        self._goal_states = goal_state
        self._flowrate_prediction_model = flowrate_prediction_model  # Model that will be used to predict the flow rate

    @property
    def initial_state(self):
        return self._initial_state

    def actions(self, state):  # actions that are available in current state
        destination_distance_pair_list = self._graph.get_edge_data(state)  # list of the destinations distance pair
        # objects for the current state
        actions_list = []

        # Extract the actions from the pairs
        for pair in destination_distance_pair_list:
            actions_list.append(pair.destination_scats_number)

        actions_list.sort()  # Sort actions in ascending order

        return actions_list  # Returns a list of SCAT sites that can be reached from the current state

    def result(self, state, action):  # The next state is the destination node
        return action

    def action_cost(self, state, action, new_state):
        distance = self._distance_between_states(state, action, new_state)  # Get the distance between states
        input_sequence_df = self._traffic_data[self._traffic_data['SCATS_Number'] == new_state].iloc[:, 3:]  # Skip Date
        # and Weekday columns

        # Use the model to make a flow rate prediction
        input_sequence_df = self._traffic_data[self._traffic_data['SCATS_Number'] == new_state]

        if input_sequence_df.empty or len(input_sequence_df) < 12:
            # Not enough history → discourage by setting flowrate low
            flowrate = 1
        else:
            # Take last 12 timesteps, full feature set
            input_sequence = input_sequence_df[['Volume_norm','Time_sin','Time_cos','Weekday_sin','Weekday_cos']].values[-12:]
            flowrate = self._flowrate_prediction_model.make_prediction(input_sequence)
        time = self._time_between_nodes(distance, flowrate, intersection_pause_time=0.00833)  # 30 seconds to pass
        # through the intersection
        return time

    def _distance_between_states(self, state, action, new_state):
        destination_cost_pair_list = self._graph.get_edge_data(state)  # Get all destinations and their distances
        # associated with current state #
        best_distance = float('inf')  # Could have multiple paths to the same destination from a given state #
        for pair in destination_cost_pair_list:  # Go through all the pairs in the cost pair list #
            if pair.destination_scats_number == new_state:  # The destination and the new state match #
                best_distance = min(best_distance, pair.distance)  # Get the best cost path to the destination #
        if best_distance != float('inf'):
            return best_distance  # Return the best estimated time #
        else:
            raise Exception('Invalid action no resulting state!')  # Edge Case: Good practice

    def _time_between_nodes(self, distance, flowrate, intersection_pause_time):
        alpha = -1.4648375  # Constants
        beta = 93.75  # Constants
        speed = (-1 * beta - math.sqrt(beta ** 2 + 4 * alpha * flowrate)) / (2 * alpha)  # Speed calculation
        if speed > 60:  # Speed limit of 60 km/h if higher is possible then revert to 60
            speed = 60
        time = distance / speed + intersection_pause_time  # Calculation for the total amount of time
        return time

    def is_goal(self, state):
        return state == self._goal_states  # if the state is the goal state