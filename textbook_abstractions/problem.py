import math
from jh_ml_models.model_deployment_abstractions.flowrate_predictor import FlowratePredictor


class Problem:
    def __init__(self, graph, initial_state, goal_state, initial_time, sequence_length, mode):
        self._graph = graph
        self._initial_state = initial_state
        self._goal_states = goal_state
        self._flowrate_predictor = FlowratePredictor(initial_time=initial_time,
                                                     sequence_length=sequence_length,
                                                     database_file_path="data/cleaned_data_base.xlsx",
                                                     mode=mode)  # Model that will be used to predict the flow rate

    @property
    def initial_state(self):
        return self._initial_state

    def actions(self, state):  # Gives the actions that are available in current state #
        destination_distance_pair_list = self._graph.get_edge_data(state)  # Gets a list of the destinations distance pair
        # objects for the current state #
        actions_list = []  # We are only interested in the actions not the cost #

        # Extract the actions from the pairs #
        for pair in destination_distance_pair_list:
            actions_list.append(pair.destination_scats_number)

        actions_list.sort()  # Sort actions in ascending order #

        return actions_list  # Returns a list of integers each representing a state (in this case the state is the scats number) that can be reached
        # from the current state #

    def result(self, state, action):  # Returns the result of performing an action on a state.
        # State is unneeded but included to remain consistent with the book #
        return action  # The action is an integer representing the destination node.
        # This can be directly returned because it is the next state #

    def action_cost(self, state, action, new_state, time_since_initial_time):
        distance = self._distance_between_states(state, action, new_state)  # Get the distance between states
        flowrate = self._flowrate_predictor.get_data(time_since_initial_time=time_since_initial_time, scats_site=new_state)  # Use the model to make a flow rate prediction #
        predicted_action_time = self._time_between_nodes(distance, flowrate, intersection_pause_time_hours=30/3600)  # Use the calculated quantities to determine the time between nodes
        return predicted_action_time

    def _distance_between_states(self, state, action, new_state):
        destination_cost_pair_list = self._graph.get_edge_data(state)  # Get all destinations and their distances
        # associated with current state #
        best_distance = float('inf')  # Could have multiple paths to the same destination from a given state #
        for pair in destination_cost_pair_list:  # Go through all the pairs in the cost pair list #
            if pair.destination_scats_number == new_state:  # The destination and the new state match #
                best_distance = min(best_distance, pair.distance)  # Get the best cost path to the destination #

        if best_distance != float('inf'):
            return best_distance  # Return the best cost #
        else:
            raise Exception('Invalid action no resulting state!')  # There is no match this should never happen #

    def _time_between_nodes(self, distance, flowrate, intersection_pause_time_hours):
        alpha = -1.4648375  # Constants
        beta = 93.75  # Constants
        speed = (-1 * beta - math.sqrt(beta ** 2 + 4 * alpha * flowrate)) / (2 * alpha)  # Speed calculation (km/hr)

        if speed > 60:  # if speed is over 60km/hr revert back to 60 km/hr
            speed = 60

        time = distance / speed + intersection_pause_time_hours  # Calculation for the total amount of time
        return time

    def is_goal(self, state):
        return state == self._goal_states  # Seeing if the state is the goal states #