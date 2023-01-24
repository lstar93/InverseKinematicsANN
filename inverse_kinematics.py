""" Planar robot inverse kinematics  """
#!/usr/bin/env python

# pylint: disable=W0511 # suppress TODOs

from math import pi, atan2, acos, degrees, asin
import numpy as np
from forward_kinematics import ForwardKinematics
from fabrik import Fabrik
from ann import ANN
from point import Point, get_distance_between, get_point_between
from plot import plot_robot

# supress printing enormous small numbers like 0.123e-16
np.set_printoptions(suppress=True)

class InverseKinematics:
    """ Inverse kinematics class """
    def __init__(self, dh_matrix, joints_lengths, workspace_limits):
        self.dh_matrix = dh_matrix
        self.joints_lengths = joints_lengths
        self.workspace_limits = workspace_limits
        self.fkine = ForwardKinematics()
        # self.first_rev_joint_point = Point([0,0,dh_matrix[0]])

    # use one of methods to compute inverse kinematics
    def ikine(self, dest_point):
        """ Calculate inverse kinematics """

class FabrikInverseKinematics(InverseKinematics):
    """ Reaching inverse kinematics using Fabrik method """
    def __init__(self, dh_matrix, joints_lengths, workspace_limits,
                 max_err = 0.001, max_iterations_num = 100):
        super().__init__(dh_matrix, joints_lengths, workspace_limits)
        self.max_err = max_err
        self.max_iterations_num = max_iterations_num

    def __fabrik(self, joints_goal_points):
        """ Calculate angles from cosine theorem """
        # theta_1 is horizontal angle and is calculated from arcus tangens
        # ensures that arm is faced into goal point direction

        theta_1 = float(atan2(joints_goal_points[3].y, joints_goal_points[3].x))

        # set rounding up to x decimal places to prevent math error
        rounding_upto = 8

        point_a = Point([0, 0, 0])
        point_b, point_c, point_d, point_e = joints_goal_points
        # todo: n-th point

        base = [point_a, point_b]
        a_to_b = get_distance_between(point_a, point_b)
        b_to_c = get_distance_between(point_b, point_c)
        c_to_d = get_distance_between(point_c, point_d)
        d_to_e = get_distance_between(point_d, point_e)
        # todo: n-th distance

        # second theta
        first_triangle = [point_a, point_c]
        a_to_c = get_distance_between(point_a, point_c)
        nominator = (pow(a_to_b,2) + pow(b_to_c,2) - pow(a_to_c,2))
        denominator = (2 * a_to_b * b_to_c)
        acos_t2 = acos(round(nominator / denominator, rounding_upto))
        if point_c.x * point_d.x < 0: # if point C and D have opposite signs
            theta_2 = (3*pi/2) - acos_t2
        else:
            theta_2 = -(pi/2 - acos_t2)

        # third theta
        second_triangle = [point_b, point_d]
        b_to_d = get_distance_between(point_b, point_d)
        nominator = (pow(b_to_c,2) + pow(c_to_d,2) - pow(b_to_d,2))
        denominator = (2 * b_to_c * c_to_d)
        acos_t3 = acos(round(nominator / denominator, rounding_upto))
        theta_3 = -(pi - acos_t3)

        # fourth theta
        third_triangle = [point_c, point_e]
        c_to_e = get_distance_between(point_c, point_e)
        nominator = (pow(c_to_d,2) + pow(d_to_e,2) - pow(c_to_e,2))
        denominator = (2 * c_to_d * d_to_e)
        acos_t4 = acos(round(nominator / denominator, rounding_upto))

        t4_point_bt = get_point_between(point_c, point_e, get_distance_between(point_c, point_e)/2)
        t4_dista = get_distance_between(point_b, t4_point_bt)
        t4_distb = get_distance_between(point_b, point_d)
        if t4_distb > t4_dista:
            theta_4 = -(pi - acos_t4)
        else:
            theta_4 = (pi - acos_t4)

        return [theta_1, theta_2, theta_3, theta_4],\
               [base, first_triangle, second_triangle, third_triangle],\
               [point_a, *joints_goal_points]

    # use one of methods to calculate inverse kinematics
    def ikine(self, dest_point):
        """ Calculate inverse kinematics """
        # Effector limits check
        if any(dp < limitv[1][0] or dp > limitv[1][1] for dp, limitv in zip(dest_point, self.workspace_limits.items())):
            raise Exception(f'Point {dest_point} is out of manipulator reach area! Limits: {self.workspace_limits}')

        # calculate theta_1 to get rid of horizontal move before FABRIK
        # theta_1 = float(atan2(dest_point[1], dest_point[0]))
        #if dest_point[1] >= 0:
        theta_1 = float(atan2(dest_point[1], dest_point[0]))
        #else:
        #    theta_1 = 2*pi + float(atan2(dest_point[1], dest_point[0]))
        self.dh_matrix[0][0] = theta_1 # replace initial theta_1

        # calculate initial xyz possition of every robot joint
        _, fk_all = self.fkine.fkine(*self.dh_matrix)

        # init_joints_positions = [Point([1,0,4]), Point([2,0,6]), Point([4,0,5]), Point([6,0,3])]
        init_joints_positions = [Point([x[0][3], x[1][3], x[2][3]]) for x in fk_all]

        # calculate joint positions using FABRIK
        fab = Fabrik(init_joints_positions,
                     self.joints_lengths,
                     self.max_err,
                     self.max_iterations_num)

        goal_joints_positions = fab.calculate(dest_point)

        # calculate manipulator angles using FABRIK
        ik_angles, _, robot = self.__fabrik(goal_joints_positions)

        return ik_angles, robot

class AnnInverseKinematics(InverseKinematics):
    """ reaching inverse kinematics using Artificial NN method """
    def __init__(self, dh_matrix, joints_lengths, workspace_limits):
        super().__init__(dh_matrix, joints_lengths, workspace_limits)
        self.ann = ANN(workspace_limits, dh_matrix)

    def load_model(self, model_name):
        """ Load model from .h5 file """
        self.ann.load_model(model_name)

    def ikine(self, dest_point, single_prediction = False):
        """ Predict thetas using neural network """
        ik_angles = []
        if single_prediction:
            ik_angles = self.ann.predict(dest_point).tolist()
        else:
            for sample in dest_point:
                ik_angles.append(self.ann.predict([sample]).tolist()[0])
            return ik_angles[0]
        return ik_angles
