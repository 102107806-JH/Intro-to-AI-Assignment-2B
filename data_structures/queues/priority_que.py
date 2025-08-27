class PriorityQueue:
    def __init__(self, key_lambda):
        self._key_lambda = key_lambda
        self._collection = []
        self._counter = 0

    def push(self, object):
        object.order_pushed_into_collection = self._counter
        self._counter += 1
        self._collection.append(object)
        self._collection = sorted(self._collection, key=self._key_lambda)

    def pop(self):
        return self._collection.pop(0)

    def is_empty(self):
        return len(self._collection) == 0

    def get_collection_states_cost(self):  # Helper function for testing #
        states = []
        for node in self._collection:
            states.append(("state: " + str(node.state), "total cost: " + str(node.total_cost),
                           "path cost: " + str(node.path_cost)))
        return states