""" Simple 3D point representation """
#!/usr/bin/env python

# pylint: disable=C0103 # suppress invalid name for x,y and z

from math import sqrt

class Point:
    """ 3D point representation"""
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __str__(self):
        return str([self.x, self.y, self.z])

    def __repr__(self):
        return str(self)

    def to_list(self):
        """ 3D point to python list """
        return [self.x, self.y, self.z]


def get_distance_between(point_a, point_b):
    """ Calculate distance between two points """
    return sqrt(pow((point_a.x - point_b.x), 2) +
                pow((point_a.y - point_b.y), 2) +
                pow((point_a.z - point_b.z), 2))

def get_point_between(start_point, end_point, distance = None):
    """ Calculate coordinates of Point between two other points \
            or coordinates of point in given distance from other point """

    # if distance is not given use middle position
    if distance is None:
        between = get_distance_between(start_point, end_point)/2

    def coords(start_point_axis, end_point_axis):
        return start_point_axis + ((distance/between)*(end_point_axis - start_point_axis))

    return Point([coords(start_point.x, end_point.x),
                  coords(start_point.y, end_point.y),
                  coords(start_point.z, end_point.z)])
