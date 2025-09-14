import unittest
from path_finding.path_finder import PathFinder
from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from datetime import datetime

class TestFunctionalTesting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
        cls.graph = testFileExtractor.extract_file_contents()
        cls.mode = "TCN"

    # Helper Methods ###################################################
    def find_solution(self, time, k_val, initial_state, goal_state, mode):
        path_finder = PathFinder(graph=TestFunctionalTesting.graph)
        return path_finder.find_paths(initial_state=initial_state, goal_state=goal_state, initial_time=time, sequence_length=12, k_val=k_val, mode=mode)

    def get_path_as_string(self, node):
        path_string = ""
        while node != None:
            path_string += f"{node.state}->"
            node = node.parent
        return path_string[:-2]  # Cut of the final '->'

    def get_path_as_list(self, node):
        path_list = []
        while node != None:
            path_list.append(node.state)
            node = node.parent
        return path_list

    def does_list_have_duplicates(self, input_list):
        for element in input_list:
            if input_list.count(element) > 1:
                return True
        return False

    def get_root_node(self, node):
        while node.parent != None:
            node = node.parent
        return node

    # Test Cases ##################################################
    def test_no_duplicate_paths(self):
        """
        Due to the nature of the path finding algorithm there should be no duplicate paths returned.
        """
        print("Running test number 1")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 2827  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
            k_val=k_val,
            initial_state=initial_state,
            goal_state=goal_state,
            mode=mode)

        encounterd_paths = {}
        path_has_already_been_seen = False
        for node in solution_nodes:
            new_path = self.get_path_as_string(node)
            path_has_already_been_seen = encounterd_paths.get(new_path, False)
            if path_has_already_been_seen:
                break
            encounterd_paths[new_path] = True
        self.assertFalse(path_has_already_been_seen)

    def test_no_duplicate_state(self):
        """
        The pathfinding algorithm performs cycle checking back to the root node of the
        tree as such there should be duplicate states in solutions.
        """
        print("Running test number 2")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 2827  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
            k_val=k_val,
            initial_state=initial_state,
            goal_state=goal_state,
            mode=mode)

        contains_duplicates = False
        for node in solution_nodes:
            path_list_for_node = self.get_path_as_list(node)
            contains_duplicates = self.does_list_have_duplicates(path_list_for_node)

        self.assertFalse(contains_duplicates)

    def test_depth_limit_works(self):
        """
        For the below origin and scats site destination it is know that there is only 1 viable (no cycle)
        path between them as such the algorithm should only produce one result and should not traverse
        through the graph indefinitely (in reality it would not be indefinite but it could potentially be an
        incredibly long time assuming each node has 2.525 neighbours on average and there are 40 nodes.
        2.525^40 = 1.23*10^16 this estimate is not perfect by any means but indicates why depth limiting
        is necessary).
        """
        print("Running test number 3")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 970  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        expected_number_of_solutions = 1
        actual_number_of_solutions = len(solution_nodes)
        self.assertEqual(expected_number_of_solutions, actual_number_of_solutions)

    def test_initial_is_goal(self):
        """
        When the initial state is the goal state are the same there should only be one solution node returned.
        """
        print("Running test number 4")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 2000  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        # There should only be one solution
        expected_number_of_solutions = 1
        actual_number_of_solutions = len(solution_nodes)
        self.assertEqual(expected_number_of_solutions, actual_number_of_solutions)

        # The solution should be the goal state
        expected_solution = initial_state
        actual_solution = solution_nodes[0].state
        self.assertEqual(expected_solution, actual_solution)

    def test_correct_number_of_solutions_returned(self):
        """
        When possible as in the situation below the number of solutions that are returned should be equal
        to the k_val
        """
        print("Running test number 5")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 4035 # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        # There should be as many solutions as the k_val specifies
        expected_number_of_solutions = k_val
        actual_number_of_solutions = len(solution_nodes)
        self.assertEqual(expected_number_of_solutions, actual_number_of_solutions)

    def test_solutions_nodes_are_returned_in_ascending_order(self):
        """
        The returned list of solution nodes should always be returned with the lowest cost
        solution first and the highest cost solution last.
        """
        print("Running test number 6")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 4035  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        # Go through all solutions and check that they are in ascending order
        order_is_correct = True
        previous_total_cost = float("-inf")
        for node in solution_nodes:
            if previous_total_cost > node.total_cost:
                order_is_correct = False
                break
            previous_total_cost = node.total_cost

        self.assertTrue(order_is_correct)

    def test_at_final_time_step_of_month(self):
        """
        This test demonstrates that predicition works at the end of the month. Demonstrating that
        the path finding uses it previous predicition values to make further predictions.
        """
        print("Running test number 7")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        # Placing the time to the end of the month
        time = time.replace(day=31)
        time = time.replace(hour=23)
        time = time.replace(minute=59)
        time = time.replace(second=59)
        k_val = 1000  # The number of solutions that you want to find. Make it high to force multiple predictions into the future
        initial_state = 2000  # Initial state from the below list
        goal_state = 3662  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        self.assertTrue(True) # No exception has been risen the program works #

    def test_at_first_time_step_of_month(self):
        """
        This test demonstrates that predicition works at the start of the month. This indicates that the values from
        the previous month can be used
        """
        print("Running test number 8")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        # Placing the time to the start of the month
        time = time.replace(day=1)
        time = time.replace(hour=0)
        time = time.replace(minute=0)
        time = time.replace(second=1)
        k_val = 1000  # The number of solutions that you want to find. Make it high to force multiple predictions into the future
        initial_state = 2000  # Initial state from the below list
        goal_state = 3662  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        self.assertTrue(True) # No exception has been risen the program works #

    def test_solutions_nodes_goals_all_correct(self):
        """
        All the returned solution nodes should have the final node as the goal.
        """
        print("Running test number 9")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 4035  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        # Go through all solutions and check all the goals are correct
        final_nodes_are_all_goal = True
        for node in solution_nodes:
            if node.state != goal_state:
                final_nodes_are_all_goal = False
                break

        self.assertTrue(final_nodes_are_all_goal)

    def test_initial_nodes_all_correct(self):
        """
        All the starting nodes should be the specified starting node.
        """
        print("Running test number 10")
        time = datetime.now()
        time = time.replace(month=8)  # We dont have access to data from current month faking that we are actually in the 8th month
        k_val = 100  # The number of solutions that you want to find
        initial_state = 2000  # Initial state from the below list
        goal_state = 4035  # Goal state from the below list
        mode = TestFunctionalTesting.mode  # LSTM, GRU or TCN
        # 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
        # 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
        # 4324, 4335, 4812, 4821,
        solution_nodes = self.find_solution(time=time,
                                            k_val=k_val,
                                            initial_state=initial_state,
                                            goal_state=goal_state,
                                            mode=mode)

        # Go through all root nodes and make sure that they are initial node
        root_nodes_are_all_initial = True
        for node in solution_nodes:
            if self.get_root_node(node).state != initial_state:
                root_nodes_are_all_initial = False
                break

        self.assertTrue(root_nodes_are_all_initial)







