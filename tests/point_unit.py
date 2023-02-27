""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115
# pylint: disable=W0612 :unused-variable

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises
from kinematics.point import Point, get_distance_between, get_point_between


class PointTest(unittest.TestCase):

    point_0 = Point([0, 0, 0])
    point_1 = Point([-2.22, 3.123, 0.002])

    # https://www.calculatorsoup.com/calculators/geometry-solids/distance-two-points.php
    distance = 3.831649

    def create(self):
        test0 = Point([0, 0, 0])
        with assert_raises(ValueError):
            test1 = Point([0, 0, 0, 1])

    def to_list(self):
        """ Point to list transofrmation """
        assert [0, 0, 0] == self.point_0
        assert [-2.22, 3.123, 0.002] == self.point_1

    def get_distance_between(self):
        """ Get distance between two points """
        ret_distance = get_distance_between(self.point_0, self.point_1)
        assert_almost_equal(self.distance, ret_distance)

    def get_point_between(self):
        """ Get point coordinates between two points """
        middle = get_point_between(self.point_0, self.point_1)
        np_middle = (np.array(self.point_0)+np.array(self.point_1))/2
        assert_almost_equal(np_middle, middle)


def point_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(PointTest('create'))
    suite.addTest(PointTest('to_list'))
    suite.addTest(PointTest('get_distance_between'))
    suite.addTest(PointTest('get_point_between'))
    return suite
