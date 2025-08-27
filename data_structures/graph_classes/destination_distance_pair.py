class DestinationDistancePair:
    def __init__(self, destination_scats_number, distance):
        self._destination_scats_number = destination_scats_number  # The destination scats number #
        self._distance = distance  # The distance to the destination #

    @property
    def destination_scats_number(self):
        return self._destination_scats_number

    @property
    def distance(self):
        return self._distance