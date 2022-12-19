#!/bin/python3

from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from datetime import datetime
from math import pi, sqrt, atan2, acos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from forward_kinematics import ForwardKinematics

import numpy as np

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

# calculate distance between two points
def get_distance_between(point_a, point_b):
    return sqrt(pow((point_a.x - point_b.x), 2) + pow((point_a.y - point_b.y), 2) + pow((point_a.z - point_b.z), 2))

# calculate coordinates of Point between two other points or coordinates of point in given distance from other point
def get_point_between(start_point, end_point, distance):
    coord_lambda = lambda start_point_axis, end_point_axis: start_point_axis + ((distance/get_distance_between(start_point, end_point))*(end_point_axis - start_point_axis))
    rx = coord_lambda(start_point.x, end_point.x)
    ry = coord_lambda(start_point.y, end_point.y)
    rz = coord_lambda(start_point.z, end_point.z)
    return Point([rx, ry, rz])

# FABRIK stands from forward and backward reaching inverse kinematics -> https://www.youtube.com/watch?v=UNoX65PRehA&feature=emb_title
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
            points_to_ret.append(get_point_between(points_to_ret[-1], next_point, distance))
        return list(reversed(points_to_ret))

    # Compute forward iteration
    def forward(self, points, start_point):
        # Compute forward joint positions -> from start position to point close to the goal position
        # Forward iteration omit first joint and begins computations from start_point
        points_to_ret = [start_point] # start point should be the first point in array
        positions = points[1:]
        distances = self.joint_distances[1:]
        for next_point, distance in zip(positions, distances):
            points_to_ret.append(get_point_between(points_to_ret[-1], next_point, distance))
        return points_to_ret

    # compute all joints positions joining backward and forward methods
    def compute(self, goal_eff_pos):
        if not all(x == len(self.init_joints_positions) for x in (len(self.init_joints_positions), len(self.joint_distances))):
            raise Exception('Input vectors should have equal lengths!')

        current_join_positions = self.init_joints_positions
        # goal_joints_positions = []
        start_point = self.init_joints_positions[0]
        goal_point = Point([x for x in goal_eff_pos])
        start_error = 1
        goal_error = 1
        step = 0

        while (((start_error > self.err_margin) or (goal_error > self.err_margin)) and (self.max_iter_num > step)):
            backward = self.backward(current_join_positions, goal_point)
            start_error = get_distance_between(backward[0], start_point)
            forward = self.forward(backward, start_point)
            goal_error = get_distance_between(forward[-1], goal_point)
            current_join_positions = forward
            # goal_joints_positions = current_join_positions
            PRINT_MSG('Iteration {} -> start position error = {}, goal position error = {}'.format(step, start_error, goal_error))
            step += 1

        return current_join_positions

# neural network IK approach
class ANN:
    def __init__(self, effector_workspace_limits, dh_matrix):
        self.effector_workspace_limits = effector_workspace_limits
        self.dh_matrix = dh_matrix

    # fit trainig data
    def fit_trainig_data(self, samples, features):
        # split data into training (70%), test and evaluation (30%)
        input, input_test_eval, output, output_test_eval = train_test_split(samples, features, test_size=0.33, random_state=105)

        # fit data using scaler
        data_skaler = MinMaxScaler()
        input_scaled = data_skaler.fit_transform(input)
        input_test_scaled = data_skaler.transform(input_test_eval)

        return np.array(input_scaled), np.array(output), np.array(input_test_scaled), np.array(output_test_eval)

    # mse custom loss function
    # def customloss(self, yTrue, yPred, no_of_samples):
    #     return (keras.backend.sum((yTrue - yPred)**2))/no_of_samples

    def train_model(self, epochs, samples, features):
        self.model = Sequential()
        data_in, data_out, data_test_in, data_test_out = self.fit_trainig_data(samples, features)

        # self.model.add(keras.layers.Dense(units=3, activation='tanh')) # x, y, z -> input layer
        self.model.add(Dense(units=720, activation='tanh')) # hidden layer 720 neurons
        self.model.add(Dense(units=1080, activation='tanh')) # hidden layer 1080 neurons
        self.model.add(Dense(units=1440, activation='tanh')) # hidden layer 1440 neurons
        self.model.add(Dense(units=2160, activation='tanh')) # hidden layer 2160 neurons
        self.model.add(Dense(units=1440, activation='tanh')) # hidden layer 1440 neurons
        self.model.add(Dense(units=1080, activation='tanh')) # hidden layer 1080 neurons
        self.model.add(Dense(units=720, activation='tanh')) # hidden layer 720 neurons
        self.model.add(Dense(units=4)) # theta1, theta2, theta3, theta4 -> output layer

        # todo: add early stopping

        self.model.compile(optimizer = Adam(learning_rate=1.0e-5), loss='mse')
        self.model.fit(data_in, data_out, validation_data=(data_test_in, data_test_out), epochs=epochs) # callbacks = [model_check]

    def predict_ik(self, position):
        return  self.model.predict(position)

    def load_model(self, model_h5):
        self.model = load_model(model_h5)
        return self.model

    def save_model(self):
        dt = datetime.now()
        # replace . with - in filename to look better
        ts_str = str(datetime.timestamp(dt)).replace('.','-')
        self.model.save('roboarm_model_'+ts_str+'.h5')

# Robo Arm inverse kinematics class
class InverseKinematics:

    def __init__(self, dh_matrix, joints_lengths, workspace_limits):
        self.dh_matrix = dh_matrix
        self.joints_lengths = joints_lengths
        self.workspace_limits = workspace_limits
        # self.first_rev_joint_point = Point([0,0,dh_matrix[0]])

    # Compute angles from cosine theorem
    # IMPORTANT: function works only for RoboArm manipulator and FABRIK method!
    def fabrik_ik(self, goal_point):
        A = Point([0, 0, 0])
        B = Point([goal_point[0].x, goal_point[0].y, goal_point[0].z])
        C = Point([goal_point[1].x, goal_point[1].y, goal_point[1].z])
        D = Point([goal_point[2].x, goal_point[2].y, goal_point[2].z])
        E = Point([goal_point[3].x, goal_point[3].y, goal_point[3].z])
        # todo: n-th point

        base = [A, B]

        AB = get_distance_between(A, B)
        BC = get_distance_between(B, C)
        CD = get_distance_between(C, D)
        DE = get_distance_between(D, E)
        # todo: n-th distance

        # theta_1 is horizontal angle and is calculated from arcus tangens 
        # ensures that arm is faced into goal point direction
        ''' view from above
        y
        /\     
        |        * gp(x,y)
        |  
        |    |__|
        |     /
        |    /
        | _ /
        |  /|
        | / |
        ----------------------> x
        '''
        theta_1 = float(atan2(goal_point[3].y, goal_point[3].x))

        # theta_2/3/4 are vertical angles, they are responsible for 
        # raching goal point vertically
        # traingles are used just to plot arm later with matplotlib

        # set rounding up to x decimal places to prevent math error
        rounding_upto = 8

        # second theta
        first_triangle = [A, C]
        AC = get_distance_between(A, C)
        nominator = (pow(AB,2) + pow(BC,2) - pow(AC,2))
        denominator = (2 * AB * BC)
        if C.x >= 0:
            theta_2 = (pi/2 - acos(round(nominator / denominator, rounding_upto))) * -1
        else:
            theta_2 = (pi + pi/2 - acos(round(nominator / denominator, rounding_upto))) # check?

        # third theta
        second_triangle = [B, D]
        BD = get_distance_between(B, D)

        nominator = (pow(BC,2) + pow(CD,2) - pow(BD,2))
        denominator = (2 * BC * CD)
        theta_3 = (pi - acos(round(nominator / denominator, rounding_upto))) * -1
        if D.x < 0:
            theta_3 = theta_3 * -1

        # fourth theta
        third_triangle = [C, E]
        CE = get_distance_between(C, E)

        nominator = (pow(CD,2) + pow(DE,2) - pow(CE,2))
        denominator = (2 * CD * DE)
        theta_4 = (pi - acos(round(nominator / denominator, rounding_upto))) * -1
        if E.x < 0:
            theta_4 = theta_4 * -1

        # todo: n-th triangle
        # ...

        return [theta_1, theta_2, theta_3, theta_4], [base, first_triangle, second_triangle, third_triangle]

    def ann_train_model(self, epochs, samples, features):
        self.ann = ANN(self.workspace_limits, self.dh_matrix)
        self.ann.train_model(epochs=epochs, input_train_data=samples, output_train_data=features) # random data

    def ann_ik(self, goal_point, train=False, model_path=None):
        ann = ANN(self.workspace_limits, self.dh_matrix)
        if train:
            ann = ANN(self.workspace_limits, self.dh_matrix)
        else:
            ann.load_model(model_path)

    # use one of methods to compute inverse kinematics
    def ikine(self, method, dest_point, max_err = 0.001, max_iterations_num = 100):

        # Effector limits check
        if any(dp < limitv[1][0] or dp > limitv[1][1] for dp, limitv in zip(dest_point, self.workspace_limits.items())):
            raise Exception("Point is out of RoboArm reach area! Limits: {}, ".format(self.workspace_limits))

        if method.lower() == "fabrik":
            # FABRIK 
            theta_1 = float(atan2(dest_point[1], dest_point[0])) # compute theta_1 to omit horizontal move in FABRIK
            self.dh_matrix[0][0] = theta_1 # replace initial theta_1

            # Compute initial xyz possition of every robot joint
            fkine = ForwardKinematics()
            _, fk_all = fkine.fkine(*self.dh_matrix)
            init_joints_positions = [Point([x[0][3], x[1][3], x[2][3]]) for x in fk_all]

            # Compute joint positions using FABRIK
            fab = Fabrik(init_joints_positions, self.joints_lengths, max_err, max_iterations_num)
            goal_joints_positions = fab.compute(dest_point)

            # Compute roboarm angles from FABRIK computed positions
            ik_angles, _ = self.fabrik_ik(goal_joints_positions)
        
            return ik_angles

        elif method.lower() == "ann":
            return None
        
        raise Exception('Unknown method!')


# Abstract class way
'''
from abc import abstractclassmethod, ABCMeta

class InverseKinematicsBase(metaclass=ABCMeta):
    @abstractclassmethod
    def ikine(self, dest_point):
        pass

class IKineFabric(InverseKinematicsBase):
    def ikine(self, dest_point):
        pass

class IKineANN(InverseKinematicsBase):
    def ikine(self, dest_point):
        pass
'''
