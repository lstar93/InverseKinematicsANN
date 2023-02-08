""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

from math import pi
from glob import glob
import unittest

from numpy.testing import assert_almost_equal
from kinematics.ann import ANN


class TestRobot():
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
    effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]}
    links_lengths = [2, 2, 2, 2]

robot = TestRobot()


class AnnTest(unittest.TestCase):

    ann = ANN(robot.effector_workspace_limits, robot.dh_matrix)

    model_name = 'tests/test_model.h5'
    model_prefix = 'tests/saved_model'

    def load_model(self):
        """ Model load test """
        assert self.ann.model is None
        model = self.ann.load_model(self.model_name)
        assert model is not None
        assert self.ann.model is not None

    def save_model(self):
        """ Model save test """
        self.ann.save_model(self.model_prefix)
        assert glob('tests/saved_model*.h5')
        assert glob('tests/saved_model*_scaler_x.bin')
        assert glob('tests/saved_model*_scaler_y.bin')

    def predict(self):
        """ IK prediction via ANN test """
        self.ann.load_model(self.model_name)
        points = [[1.0, 2.1, 3.0], [-1.567, 2.22, -3.123], [1.02, 3.33, 4.99]]
        output = [[1.1388013362884521, 1.9391953945159912, -1.544973611831665, -1.3534488677978516],
        [1.5541561841964722, 0.5132879614830017, -2.0762546062469482, -0.5687951445579529],
        [1.286367416381836, 1.4932732582092285, -0.685073971748352, -1.0487391948699951]]

        predicted = self.ann.predict(points)
        assert_almost_equal(predicted, output)

        # this should go to ann inverse kinematics class test :)
        # with assert_raises(OutOfRobotReachException):
        #     points = [[1.0, 22.1, 3.0]]
        #     predicted = self.ann.predict(points)

def ann_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(AnnTest('load_model'))
    suite.addTest(AnnTest('save_model'))
    suite.addTest(AnnTest('predict'))
    return suite
