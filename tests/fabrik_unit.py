""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

from math import pi
import unittest

from numpy.testing import assert_array_almost_equal
from kinematics.fabrik import Fabrik
from kinematics.point import Point
from kinematics.forward import ForwardKinematics

class TestRobot():
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
    effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]}
    links_lengths = [2, 2, 2, 2]

robot = TestRobot()


class FabrikTest(unittest.TestCase):

    fabrik = Fabrik(robot.links_lengths)
    fkine = ForwardKinematics(robot.dh_matrix)

    def calculate(self):
        """ IK prediction via Fabrik test """
        destination = [1,2,3]
        output = [[0.0, 0.0, 2.0],
        [-0.3524468346566213, -0.7083832214867447, 3.836838163871981],
        [0.472013795834034, 0.9406164237214534, 4.612121878955813],
        [1.0000000035582093, 2.0000000071394073, 2.999999989135574]]

        _, fkall = self.fkine.fkine(robot.dh_matrix[0])
        starting_positions = [Point([fka[0,3], fka[1,3], fka[2,3]]) for fka in fkall]

        calculated = self.fabrik.calculate(starting_positions, destination)
        assert_array_almost_equal(calculated[3].to_list(), output[3])


def fabrik_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(FabrikTest('calculate'))
    return suite
