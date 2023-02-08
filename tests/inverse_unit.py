""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import unittest
import numpy as np
from robot.robot import robot
from numpy.testing import assert_almost_equal
from kinematics.inverse import FabrikInverseKinematics, AnnInverseKinematics


class InverseKinematicsFabrikTest(unittest.TestCase):

    ikine = FabrikInverseKinematics(robot.dh_matrix,
                                    robot.links_lengths,
                                    robot.effector_workspace_limits)

    def inverse_kine(self):
        """ Inverse kinematics using fabrik test """


class InverseKinematicsAnnTest(unittest.TestCase):

    model_name = 'tests/test_model.h5'
    ikine = AnnInverseKinematics(robot.dh_matrix, 
                                 robot.links_lengths, 
                                 robot.effector_workspace_limits)

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.ikine.load_model(self.model_name)

    def inverse_kine(self):
        """ Inverse kinematics using ANN test """


def ikine_fabrik_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(InverseKinematicsFabrikTest('inverse_kine_fabrik'))
    suite.addTest(InverseKinematicsAnnTest('inverse_kine'))
    return suite
