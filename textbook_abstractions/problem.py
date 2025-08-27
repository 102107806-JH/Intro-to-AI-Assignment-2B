class Problem:
    def __init__(self, graph, initial_state, goal_state):  # Takes an extracted text file object  #
        self._graph = graph
        self._initial_state = initial_state
        self._goal_states = goal_state

    @property
    def initial_state(self):
        return self._initial_state

    def actions(self, state):  # Gives the actions that are available in current state #
        destination_distance_pair_list = self._graph.get_edge_data(state)  # Gets a list of the destination cost pair
        # objects for the current state #
        actions_list = []  # We are only interested in the actions not the cost #

        # Extract the actions from the pairs #
        for pair in destination_distance_pair_list:
            actions_list.append(pair.destination_scats_number)

        actions_list.sort()  # Sort actions in ascending order #

        return actions_list  # Returns a list of integers each representing a state that can be reached
        # from the current state #

    def result(self, state, action):  # Returns the result of performing an action on a state.
        # State is unneeded but included to remain consistent with the book #
        return action  # The action is an integer representing the destination node.
        # This can be directly returned because it is the next state #

    def action_cost(self, state, action, new_state):
        destination_cost_pair_list = self._graph.get_edge_data(state)  # Get all destinations and their costs
        # associated with current state #
        best_cost = float('inf')  # Could have multiple paths to the same destination from a given state #
        for pair in destination_cost_pair_list:  # Go through all the pairs in the cost pair list #
            if pair.destination_scats_number == new_state:  # The destination and the new state match #
                best_cost = min(best_cost, pair.distance)  # Get the best cost path to the destination #

        if best_cost != float('inf'):
            return best_cost  # Return the best cost #
        else:
            raise Exception('Invalid action no resulting state!')  # There is no match this should never happen #

    def is_goal(self, state):
        return state == self._goal_states  # Seeing if the state is in the goal states #
