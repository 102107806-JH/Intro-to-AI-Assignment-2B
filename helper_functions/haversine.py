import math

# The following code is based off the pseudo code from : https://www.movable-type.co.uk/scripts/latlong.html #
def haversine(long_1, lat_1, long_2, lat_2):
    # Convert all measurements into radians
    long_1_rad = long_1 * math.pi / 180
    lat_1_rad = lat_1 * math.pi / 180
    long_2_rad = long_2 * math.pi / 180
    lat_2_rad = lat_2 * math.pi / 180

    # Calculate deltas #
    delta_long_rad = long_2_rad - long_1_rad
    delta_lat_rad = lat_2_rad - lat_1_rad

    # Intermediate value calculation
    a = math.sin(delta_lat_rad / 2) ** 2 + math.cos(lat_1_rad) * math.cos(lat_2_rad) * math.sin(delta_long_rad / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of the earth in km

    # Distance calculation
    d = r * c

    return d

