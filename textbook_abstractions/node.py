class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, total_cost=0):
        self._state = state  # The node state #
        self._parent = parent  # Reference to the parent node #
        self._action = action  # The Action that was used to get to the state of this node #
        self._path_cost = path_cost  # The path cost of getting to this node (In this node version if there is a heuristic it will be added outside the function)#
        self._total_cost = total_cost  # The total cost at this node #
        self._order_pushed_into_collection = None  # The order which the node was pushed onto the collection #


    @property
    def state(self):
        return self._state

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def action(self):
        return self._action

    @property
    def path_cost(self):
        return self._path_cost

    @property
    def total_cost(self):
        return self._total_cost

    @property
    def order_pushed_into_collection(self):
        return self._order_pushed_into_collection

    @order_pushed_into_collection.setter
    def order_pushed_into_collection(self, order_pushed_into_collection):
        self._order_pushed_into_collection = order_pushed_into_collection