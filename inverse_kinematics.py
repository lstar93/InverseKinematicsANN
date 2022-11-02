#!/bin/python3

from enum import Enum
from keras.models import load_model
from datetime import datetime
import numpy as np
from math import pi, sqrt, atan2, acos
import sklearn
from sklearn import model_selection
import keras
import tensorflow as tf

from forward_kinematics import ForwardKinematics

# supress printing enormous small numbers like 0.123e-16
np.set_printoptions(suppress=True)

# Keep some prints, but show them only if necessary
VERBOSE = False
def PRINT_MSG(msg, verbose=VERBOSE):
    if verbose:
        print(msg)

def pow(arg, p):
    return float(arg ** p)

class Point:
    x = 0.0
    y = 0.0
    z = 0.0

    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __str__(self):
        return str([self.x, self.y, self.z])

    def __repr__(self):
        return str(self)

    def to_list(self):
        return [self.x, self.y, self.z]

def distance_between_points(point_a, point_b):
    return sqrt(pow((point_a.x - point_b.x), 2) + pow((point_a.y - point_b.y), 2) + pow((point_a.z - point_b.z), 2))

def get_point_between_points(start_point, end_point, distance):
    rx = start_point.x + ((distance/distance_between_points(start_point, end_point))*(end_point.x - start_point.x))
    ry = start_point.y + ((distance/distance_between_points(start_point, end_point))*(end_point.y - start_point.y))
    rz = start_point.z + ((distance/distance_between_points(start_point, end_point))*(end_point.z - start_point.z))
    return Point([rx, ry, rz])

# FABRIK stands from forward and backward reaching inverse kinematics -> https://www.youtube.com/watch?v=UNoX65PRehA&feature=emb_title
# TODO: add angles limits

class Fabrik:

    def __init__(self, init_joints_positions, joint_distances, err_margin = 0.001, max_iter_num = 100):
        self.joint_distances = joint_distances
        self.init_joints_positions = init_joints_positions
        self.err_margin = err_margin
        self.max_iter_num = max_iter_num
        
    # Compute backward iteration
    def backward(self, points, goal_point):
        # Compute backward joint positions -> from goal position to point close to the start joint
        # Backward iteration omit last joint and begins computations from goal_point
        points_to_ret = [goal_point] # goal point should be the last point in array
        positions = list(reversed(points[:-1]))
        distances = list(reversed(self.joint_distances[:-1]))
        for next_point, distance in zip(positions, distances):
            points_to_ret.append(get_point_between_points(points_to_ret[-1], next_point, distance))
        return list(reversed(points_to_ret))

    # Compute forward iteration
    def forward(self, points, start_point):
        # Compute forward joint positions -> from start position to point close to the goal position
        # Forward iteration omit first joint and begins computations from start_point
        points_to_ret = [start_point] # start point should be the first point in array
        positions = points[1:]
        distances = self.joint_distances[1:]
        for next_point, distance in zip(positions, distances):
            points_to_ret.append(get_point_between_points(points_to_ret[-1], next_point, distance))
        return points_to_ret

    def compute_goal_joints_positions(self, goal_eff_pos):
        if not all(x == len(self.init_joints_positions) for x in (len(self.init_joints_positions), len(self.joint_distances))):
            raise Exception('Input vectors should have equal lengths!')

        current_join_positions = self.init_joints_positions
        goal_joints_positions = []
        start_point = self.init_joints_positions[0]
        goal_point = Point([x for x in goal_eff_pos])
        start_error = 1
        goal_error = 1
        iter_cnt = 0

        while (((start_error > self.err_margin) or (goal_error > self.err_margin)) and (self.max_iter_num > iter_cnt)):
            retb = self.backward(current_join_positions, goal_point)
            start_error = distance_between_points(retb[0], start_point)
            retf = self.forward(retb, start_point)
            goal_error = distance_between_points(retf[-1], goal_point)
            current_join_positions = retf
            goal_joints_positions = current_join_positions
            PRINT_MSG('Iteration {} -> start position error = {}, goal position error = {}'.format(iter_cnt, start_error, goal_error))
            iter_cnt += 1

        # if verbose and not len(goal_joints_positions) == 0:
        #     base_point = Point([0, 0, 0])
        #     base = [base_point, goal_joints_positions[0]]
        #     plot_robot([base, self.init_joints_positions, goal_joints_positions], [base_point, goal_point, start_point])

        return goal_joints_positions

# neural network IK approach
class ANN:
    effector_workspace_limits = {}
    angles_limits = {}
    dh_matrix = []
    model = None

    # data scalers
    data_skaler = sklearn.preprocessing.MinMaxScaler()
    # out_data_skaler = sklearn.preprocessing.MinMaxScaler()

    def __init__(self, angles_limits, effector_workspace_limits, dh_matrix):
        self.angles_limits = angles_limits
        self.effector_workspace_limits = effector_workspace_limits
        self.dh_matrix = dh_matrix

    # fit trainig data
    def fit_trainig_data(self, samples, features):
        # split data into training (70%), test and evaluation (30%)
        input, input_test_eval, output, output_test_eval = model_selection.train_test_split(samples, features, test_size=0.3, random_state=42)

        # fit data using scaler
        self.data_skaler.fit(input)
        input_scaled = self.data_skaler.fit_transform(input)
        output_scaled = output # self.out_data_skaler.fit_transform(output)
        input_test_scaled = self.data_skaler.fit_transform(input_test_eval)
        output_test_scaled = output_test_eval # self.out_data_skaler.fit_transform(output_test_eval)

        return np.array(input_scaled), np.array(output_scaled), np.array(input_test_scaled), np.array(output_test_scaled)

    def customloss(self, yTrue, yPred, no_of_samples):
        return (keras.backend.sum((yTrue - yPred)**2))/no_of_samples

    def train_model(self, epochs, input_train_data, output_train_data):
        self.model = keras.Sequential()
        data_in, data_out, data_test_in, data_test_out = self.fit_trainig_data(input_train_data, output_train_data)

        # self.model.add(keras.layers.Dense(units=3, activation='tanh')) # x, y, z -> input layer
        self.model.add(keras.layers.Dense(units=720, activation='tanh')) # hidden layer 720 neurons
        self.model.add(keras.layers.Dense(units=1080, activation='tanh')) # hidden layer 1080 neurons
        self.model.add(keras.layers.Dense(units=1440, activation='tanh')) # hidden layer 1440 neurons
        self.model.add(keras.layers.Dense(units=2160, activation='tanh')) # hidden layer 2160 neurons
        self.model.add(keras.layers.Dense(units=1440, activation='tanh')) # hidden layer 1440 neurons
        self.model.add(keras.layers.Dense(units=1080, activation='tanh')) # hidden layer 1080 neurons
        self.model.add(keras.layers.Dense(units=720, activation='tanh')) # hidden layer 720 neurons
        self.model.add(keras.layers.Dense(units=4)) # theta1, theta2, theta3, theta4 -> output layer

        # model_check = tf.keras.callbacks.ModelCheckpoint(filepath = 'net_weights.h5', verbose = True, save_best_only = True)
        adam_opt = tf.keras.optimizers.Adam(lr=1.0e-5)
        self.model.compile(optimizer=adam_opt, loss='mse')
        self.model.fit(data_in, data_out, validation_data=(data_test_in, data_test_out), epochs=epochs) # callbacks = [model_check]

    def predict_ik(self, position):
        arraynp = np.array(position)
        position_scaled = self.data_skaler.fit_transform(arraynp)
        predictions = self.model.predict(position_scaled)
        # self.out_data_skaler.inverse_transform(predictions)
        return predictions

    def load_model(self, model_h5):
        self.model = load_model(model_h5)
        return self.model

    def save_model(self):
        # Getting the current date and time
        dt = datetime.now()
        # getting the timestamp as str with . replaced with - to look nicer
        ts_str = str(datetime.timestamp(dt)).replace('.','-')
        self.model.save('roboarm_model_'+ts_str+'.h5')

# Robo Arm inverse kinematics class
class InverseKinematics:

    def __init__(self, dh_matrix, joints_lengths, workspace_limits):
        self.dh_matrix = dh_matrix
        self.joints_lengths = joints_lengths
        self.workspace_limits = workspace_limits
        self.first_rev_joint_point = Point([0, 0, dh_matrix[1][0]]) # take first joint vertical length

    # Compute angles from cosine theorem
    # IMPORTANT: function works only for RoboArm manipulator and FABRIK method!
    def fabrik_ik(self, goal_point):
        A = Point([0, 0, 0])
        B = Point([goal_point[0].x, goal_point[0].y, goal_point[0].z])
        C = Point([goal_point[1].x, goal_point[1].y, goal_point[1].z])
        D = Point([goal_point[2].x, goal_point[2].y, goal_point[2].z])
        E = Point([goal_point[3].x, goal_point[3].y, goal_point[3].z])

        base = [A, B]

        AB = distance_between_points(A, B)
        BC = distance_between_points(B, C)
        CD = distance_between_points(C, D)
        DE = distance_between_points(D, E)

        # first triangle
        first_triangle = [A, C]
        AC = distance_between_points(A, C)
        if C.x >= 0:
            theta_2 = (pi/2 - acos((pow(AB,2) + pow(BC,2) - pow(AC,2)) / (2 * AB * BC))) * -1
        else:
            theta_2 = (pi + pi/2 - acos((pow(AB,2) + pow(BC,2) - pow(AC,2)) / (2 * AB * BC)))

        # second triangle
        second_triangle = [B, D]
        BD = distance_between_points(B, D)
        theta_3 = (pi - acos((pow(BC,2) + pow(CD,2) - pow(BD,2)) / (2 * BC * CD))) * -1
        if D.x < 0:
            theta_3 = theta_3 * -1

        # third triangle
        third_triangle = [C, E]
        CE = distance_between_points(C, E)
        theta_4 = (pi - acos((pow(CD,2) + pow(DE,2) - pow(CE,2)) / (2 * CD * DE))) * -1
        if E.x < 0:
            theta_4 = theta_4 * -1

        theta_1 = float(atan2(goal_point[3].y, goal_point[3].x))

        return [theta_1, theta_2, theta_3, theta_4], [base, first_triangle, second_triangle, third_triangle]

    def ann_ik(self, gp):
        pass

    # use one of methods to compute inverse kinematics
    def compute_roboarm_ik(self, method, dest_point, max_err = 0.001, max_iterations_num = 100):
        # Some basic limits check
        if any(dp < limitv[1][0] or dp > limitv[1][1] for dp, limitv in zip(dest_point, self.workspace_limits.items())):
            raise Exception("Point is out of RoboArm reach area! Limits: {}, ".format(self.workspace_limits))

        effector_reach_limit = self.workspace_limits['x_limits'][1]

        # Roboarm reach distance check
        # TODO: remove this and assume that first joint can move vertically!!!
        if distance_between_points(self.first_rev_joint_point, Point(dest_point)) > effector_reach_limit: 
            raise Exception("Point is out of RoboArm reach area! Reach limit is {}, but the distance to point is {}".format(effector_reach_limit, distance_between_points(self.first_rev_joint_point, Point(dest_point))))

        if method.lower() == "fabrik":
            # FABRIK 
            theta_1 = float(atan2(dest_point[1], dest_point[0])) # compute theta_1 to omit horizontal move in FABRIK
            self.dh_matrix[0][0] = theta_1 # replace initial theta_1

            # Compute initial xyz possition of every robot joint
            fkine = ForwardKinematics()
            _, fk_all = fkine.forward_kinematics(*self.dh_matrix)
            init_joints_positions = [Point([x[0][3], x[1][3], x[2][3]]) for x in fk_all]

            # Compute joint positions using FABRIK
            fab = Fabrik(init_joints_positions, self.joints_lengths, max_err, max_iterations_num)
            goal_joints_positions = fab.compute_goal_joints_positions(dest_point)

            # Compute roboarm angles from FABRIK computed positions
            ik_angles, joints_triangles = self.fabrik_ik(goal_joints_positions)
            
            # print robot arm
            # if verbose:
            #     plot_roboarm([*joints_triangles, init_joints_positions, goal_joints_positions], [init_joints_positions[0], goal_joints_positions[-1]])
        
            return ik_angles
        
        raise Exception('Unknown method!')