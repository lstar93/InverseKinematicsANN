""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import unittest
import numpy as np
from robot.robot import robot
from numpy.testing import assert_almost_equal
from kinematics.forward import ForwardKinematics


class ForwardKinematicsTest(unittest.TestCase):

    fkine = ForwardKinematics(robot.dh_matrix)

    def forward_kine(self):
        """ Forward kinematics test """

def fwkine_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(ForwardKinematicsTest('forward_kine'))
    return suite
