""" Simple 3D point representation """
#!/usr/bin/env python

# pylint: disable=C0103 # suppress invalid name for x, y and z

from math import sqrt
from numpy import array


class Point(list):
    """ 3D point representation"""
    def __init__(self, xyz: list):
        if array(xyz).shape != (3,):
            raise ValueError(f'3D Point input shape should be (3,) not {array(xyz).shape}')
        super().__init__(xyz)
        self.x, self.y, self.z = xyz

    def __str__(self):
        return str(f'Point{self.x, self.y, self.z}')

    def __repr__(self):
        return f'<Point at 0x{id(self):x}, x={self.x}, y={self.y}, z={self.z}>'


def get_distance_between(point_a, point_b):
    """ Calculate distance between two points """
    return sqrt(pow((point_a.x - point_b.x), 2) +
                pow((point_a.y - point_b.y), 2) +
                pow((point_a.z - point_b.z), 2))


def get_point_between(start_point, end_point, distance = None):
    """ Calculate coordinates of Point in given distance or between two other points"""
    # default distance is position exactlly in the middle between points
    if distance is None:
        distance = get_distance_between(start_point, end_point)/2

    def coords(start_point_axis, end_point_axis):
        return start_point_axis + \
            ((distance/get_distance_between(start_point, end_point)) * \
            (end_point_axis - start_point_axis))

    return Point([coords(start_point.x, end_point.x),
                  coords(start_point.y, end_point.y),
                  coords(start_point.z, end_point.z)])
