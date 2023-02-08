""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import unittest
from numpy.testing import assert_almost_equal, assert_raises
from robot.robot import SixDOFRobot, OutOfRobotReachException
from kinematics.inverse import FabrikInverseKinematics, AnnInverseKinematics

robot = SixDOFRobot()

class InverseKinematicsFabrikTest(unittest.TestCase):

    ikine = FabrikInverseKinematics(robot.dh_matrix,
                                    robot.links_lengths,
                                    robot.effector_workspace_limits)

    def inverse_kine(self):
        """ Inverse kinematics using fabrik test """
        points = [[1.0, 2.1, 3.0], [1.567, 2.22, -2.123], [1.02, 3.33, 4.99]]
        output = [[1.1263771168937977, 1.95663870779144, -1.581170282866297, -1.2914981807424972],
        [0.9561510602151175, -0.1334947854494175, -1.441291844752837, 0.38467252287989595],
        [1.2735640189772053, 1.4953811089376177, -0.6880936114216039, -1.03376967052818]]

        predicted = self.ikine.ikine(points)

        assert_almost_equal(predicted, output)

        with assert_raises(OutOfRobotReachException):
            points = [[1.0, 2.1, 3.0], [1.567, 2.22, -3.123], [1.02, 3.33, 4.99]]
            predicted = self.ikine.ikine(points)


class InverseKinematicsAnnTest(unittest.TestCase):

    model_name = 'tests/test_model.h5'
    ikine = AnnInverseKinematics(robot.dh_matrix,
                                 robot.links_lengths,
                                 robot.effector_workspace_limits)

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.ikine.load_model(self.model_name)

    def inverse_kine(self):
        """ Inverse kinematics using ANN test """

        self.ikine.load_model(self.model_name)
        points = [[1.0, 2.1, 3.0], [1.567, 2.22, -2.123], [1.02, 3.33, 4.99]]
        output = [[1.1388013362884521, 1.9391953945159912, -1.544973611831665, -1.3534488677978516],
        [0.9609791040420532, -0.16555732488632202, -1.4587481021881104, -0.4864034056663513],
        [1.286367416381836, 1.4932732582092285, -0.685073971748352, -1.0487391948699951]]

        predicted = self.ikine.ikine(points)
        assert_almost_equal(predicted, output)

        with assert_raises(OutOfRobotReachException):
            points = [[1.0, 2.1, 3.0], [1.567, 2.22, -3.123], [1.02, 3.33, 4.99]]
            predicted = self.ikine.ikine(points)


def ikine_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(InverseKinematicsFabrikTest('inverse_kine'))
    suite.addTest(InverseKinematicsAnnTest('inverse_kine'))
    return suite
