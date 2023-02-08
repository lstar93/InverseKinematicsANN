""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import unittest
from numpy.testing import assert_array_almost_equal
from robot.robot import SixDOFRobot
from kinematics.forward import ForwardKinematics

robot = SixDOFRobot()

class ForwardKinematicsTest(unittest.TestCase):

    fkine = ForwardKinematics(robot.dh_matrix)
    destination_points = [[1.34542,2.99821,3.67401],
                          [0.01333,-3.72111,-1.09902],
                          [3.95444, -1.00112, 1.00378]]

    angles = [[1.1489898108341745, 1.6426609377538854, -1.2027772444264693, -1.0663073873609727],
            [-1.5672140776862065, 0.2433182869870163, -1.3760689820099818, 0.0465569704233757],
            [-0.24795388218721454, 0.9644220067435634, -1.5389903144536021, -0.3143083371860276]]

    def forward_kine(self):
        """ Forward kinematics test """
        for angle, dest in zip(self.angles, self.destination_points):
            fka, _ = self.fkine.fkine(angle)
            positions = [fka[0,3], fka[1,3], fka[2,3]]
            assert_array_almost_equal(dest, positions, decimal=4)


def fwkine_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(ForwardKinematicsTest('forward_kine'))
    return suite
