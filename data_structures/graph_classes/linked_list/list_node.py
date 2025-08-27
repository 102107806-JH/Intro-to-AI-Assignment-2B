class ListNode:
    def __init__(self, data):
        self._data = data  # The data of the linked list node #
        self._next = None  # The link to the next node #

    @property
    def data(self):  # Get the data of the node #
        return self._data

    @property
    def next(self):  # Get the next node #
        return self._next

    @next.setter
    def next(self, next):  # Set the next node #
        self._next = next




