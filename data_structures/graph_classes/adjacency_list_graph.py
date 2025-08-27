from data_structures.graph_classes.vertex import Vertex
from data_structures.graph_classes.destination_distance_pair import DestinationDistancePair
from helper_functions.haversine import haversine

class AdjacencyListGraph:
    def __init__(self):
        self._adjacency_list = []  # Stores all the graph vertices #
        self._scats_to_idx_dict = {} # Stores the current scats number as the key and the index within the adjacency list as the value

    def insert_vertex(self, scats_number, long, lat):  # Insert another vertex into adjacency list #
        self._scats_to_idx_dict[scats_number] = len(self._adjacency_list)  # Length of the list before appending is the index of the incoming scats number vertex in the list #
        self._adjacency_list.append(Vertex(scats_number, long, lat))  # Calling the vertex constructor #

    def insert_edge_into_vertex(self, origin_scats_number, destination_scats_number):  # Insert another edge for the origin scats number #
        origin_vertex = self.scats_number_to_vertex(origin_scats_number)  # Obtain the origin vertex specified by the origin scats number #
        destination_vertex = self.scats_number_to_vertex(destination_scats_number)  # Obtain the destination vertex specified by the destination scats number #
        distance = haversine(long_1=origin_vertex.long, lat_1=origin_vertex.lat, long_2=destination_vertex.long, lat_2=destination_vertex.lat)  # Get the distance between the two points #
        origin_vertex.add_edge(DestinationDistancePair(destination_scats_number, distance))  # Destination cost pair constructor #

    def get_edge_data(self, scats_number):  # Extract the edge data for a given origin vertex #
        vertex = self.scats_number_to_vertex(scats_number)  # Get the vertex corresponding to the scats number#
        return vertex.edges  # Returns a list of the edges #

    def scats_number_to_vertex(self, scats_number):
        vertex_idx = self._scats_to_idx_dict.get(scats_number, None)
        if vertex_idx is None:
            raise Exception("The given scats number: " + scats_number + " has no corresponding vertex in the graph")
        else:
            return self._adjacency_list[vertex_idx]

        # Len gives the number of graph nodes. The subtracting 1 gives the diameter #
    def get_graph_diameter(self):
        return len(self._adjacency_list) - 1