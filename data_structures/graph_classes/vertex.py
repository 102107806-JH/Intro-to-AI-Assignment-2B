from data_structures.graph_classes.linked_list.singly_linked_list import SinglyLinkedList


class Vertex:
    def __init__(self, scats_number, long, lat):
        self._scats_number = scats_number  # The scats (number of the vertex) #
        self._long = long  # The longitude of the vertex #
        self._lat = lat  # The latitude of the vertex #
        self._edges = SinglyLinkedList()  # A linked list that will store all the destination distance pairs of all the edges #

    def add_edge(self, edge):
        self._edges.append(edge)  # Append an edge to the singularly linked list #

    @property
    def edges(self):  # Getting the edges in list format #
        # The below code extracts the data from the linked list and places it in list
        current_node = self._edges.head  # Set the current node to the head of the linked list #
        edges_list = []  # List that will store the extracted data from the linked list #

        # Traverse entire linked list extracting all the data and placing it in the edges list
        while current_node is not None:
            edges_list.append(current_node.data)
            current_node = current_node.next

        return edges_list

    @property
    def scats_number(self):
        return self._scats_number

    @property
    def lat(self):
        return self._lat

    @property
    def long(self):
        return self._long