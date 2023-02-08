""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import unittest
from kinematics.point import Point


class PointTest(unittest.TestCase):

    point_0 = Point([0, 0, 0])
    point_1 = Point([-2.22, 3.123, 0.002])

    def to_list(self):
        """ Point to list transofrmation test """
        assert [0, 0, 0] == self.point_0.to_list()
        assert [-2.22, 3.123, 0.002] == self.point_1.to_list()

def point_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(PointTest('to_list'))
    return suite
