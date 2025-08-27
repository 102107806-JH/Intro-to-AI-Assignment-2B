class Node:
    def __init__(self, state, parent=None, action=None, total_cost=0, path_cost=0):
        self._state = state
        self._parent = parent
        self._action = action
        self._total_cost = total_cost
        self._path_cost = path_cost
        self._order_pushed_into_collection = None


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