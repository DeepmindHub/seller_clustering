import math


def degree_to_radian(lat, long):
    rad_lat = math.pi*lat/180
    rad_long = math.pi*long/180
    return (rad_lat, rad_long)


def computeHaversineDistance(startLatitude, startLongitude, endLatitude, endLongitude):

    # if (int(endLatitude) == 0 or int(endLongitude) == 0):
    #   distance = 1
    #   return distance

    startLatitude, startLongitude = degree_to_radian(startLatitude, startLongitude)
    endLatitude, endLongitude = degree_to_radian(endLatitude, endLongitude)

    earth_radius = 6371

    try:
        distance = math.acos(min(1, math.sin(startLatitude) * math.sin(endLatitude) + math.cos(startLatitude)
                                 * math.cos(endLatitude) * math.cos(startLongitude-endLongitude))) * earth_radius
        return distance
    except Exception, e:
        # distance = 1
        print startLatitude, startLongitude, endLatitude, endLongitude
        print e
        raise e
