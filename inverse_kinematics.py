""" Planar robot inverse kinematics  """
#!/usr/bin/env python

# pylint: disable=W0511 # suppress TODOs

from datetime import datetime
from math import pi, sqrt, atan2, acos
from joblib import dump, load
import numpy as np
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from keras import activations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from forward_kinematics import ForwardKinematics

# supress printing enormous small numbers like 0.123e-16
np.set_printoptions(suppress=True)

# Keep some prints, but show them only if necessary
VERBOSE = False
def print_debug_msg(msg, verbose=VERBOSE):
    """ Debug printer """
    if verbose:
        print(msg)


class Point:
    """ 3D point representation"""
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __str__(self):
        return str([self.x, self.y, self.z])

    def __repr__(self):
        return str(self)

    def to_list(self):
        """ 3D point to python list """
        return [self.x, self.y, self.z]


def get_distance_between(point_a, point_b):
    """ Compute distance between two points """
    return sqrt(pow((point_a.x - point_b.x), 2) + pow((point_a.y - point_b.y), 2) + pow((point_a.z - point_b.z), 2))

def get_point_between(start_point, end_point, distance):
    """ Compute coordinates of Point between two other points \
            or coordinates of point in given distance from other point """
    def coords(start_point_axis, end_point_axis):
        return start_point_axis + ((distance/get_distance_between(start_point, end_point))*(end_point_axis - start_point_axis))
    return Point([coords(start_point.x, end_point.x), coords(start_point.y, end_point.y), coords(start_point.z, end_point.z)])


class Fabrik:
    """ FABRIK stands from forward and backward reaching inverse kinematics ->\
            https://www.youtube.com/watch?v=UNoX65PRehA&feature=emb_title """
    def __init__(self, init_joints_positions, joint_distances, err_margin = 0.001, max_iter_num = 100):
        self.joint_distances = joint_distances
        self.init_joints_positions = init_joints_positions
        self.err_margin = err_margin
        self.max_iter_num = max_iter_num

    # Compute backward iteration
    def backward(self, points, goal_point):
        """ Compute backward joint positions -> \
            from goal position to point close to the start joint """
        points_to_ret = [goal_point] # goal point should be the last point in array
        positions = list(reversed(points[:-1]))
        distances = list(reversed(self.joint_distances[:-1]))
        for next_point, distance in zip(positions, distances):
            points_to_ret.append(get_point_between(points_to_ret[-1], next_point, distance))
        return list(reversed(points_to_ret))

    # Compute forward iteration
    def forward(self, points, start_point):
        """ Calculate forward joint positions -> \
            from start position to point close to the goal position """
        points_to_ret = [start_point] # start point should be the first point in array
        positions = points[1:]
        distances = self.joint_distances[1:]
        for next_point, distance in zip(positions, distances):
            points_to_ret.append(get_point_between(points_to_ret[-1], next_point, distance))
        return points_to_ret

    def calculate(self, goal_eff_pos):
        """ Calculate all joints positions joining backward and forward methods """
        if not all(x == len(self.init_joints_positions) for x in (len(self.init_joints_positions), len(self.joint_distances))):
            raise Exception('Input vectors should have equal lengths!')

        current_join_positions = self.init_joints_positions
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
            print_debug_msg(f'Iteration {step} -> start position error = {start_error}, '\
                        'goal position error = {goal_error}')
            step += 1

        return current_join_positions


class ANN:
    """ ANN class to neural network IK approach """
    def __init__(self, effector_workspace_limits, dh_matrix):
        self.effector_workspace_limits = effector_workspace_limits
        self.dh_matrix = dh_matrix
        self.model = Sequential()
        self.x_data_skaler = StandardScaler()
        self.y_data_skaler = StandardScaler()

    def fit_trainig_data(self, samples, features):
        """ Split training/test (70/30) data and use MinMaxScaler to scale it """
        x_train, x_test, y_train, y_test = \
            train_test_split(samples, features, test_size=0.33, random_state=42)

        x_train = self.x_data_skaler.fit_transform(x_train)
        x_test = self.x_data_skaler.transform(x_test)

        y_train = self.y_data_skaler.fit_transform(y_train)
        y_test = self.y_data_skaler.transform(y_test)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    def train_model(self, epochs, samples, features):
        """ Train ANN Sequential model """
        data_in, data_out, data_test_in, data_test_out = self.fit_trainig_data(samples, features)

        self.model.add(Input(shape=(3,))) # Input layer, 3 input variables

        net_shape = [
                (12, 500, activations.tanh)
            ]

        for shape in net_shape:
            for _ in range(shape[0]):
                self.model.add(Dense(units=shape[1], activation=shape[2])) # hidden layer

        self.model.add(Dense(units=4)) # theta1, theta2, theta3, theta4 -> output layer

        early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

        self.model.compile(optimizer = Adam(learning_rate=1.0e-5), loss='mse')

        self.model.fit(
                        data_in, data_out,
                        validation_data = (data_test_in, data_test_out),
                        epochs = epochs,
                        callbacks = [early_stopping],
                        # batch_size=64
                      )

    def predict_ik(self, position):
        """ Use trained ANN to predict joint angles """
        predictions = self.y_data_skaler.inverse_transform(
            self.model.predict(self.x_data_skaler.transform(position))
        )

        return predictions

    def load_model(self, model_h5):
        """ Load model from file """
        self.model = load_model(model_h5)
        modelname = model_h5[:-3]
        # load scalers for this model
        self.x_data_skaler = load(f'{modelname}_scaler_x.bin')
        self.y_data_skaler = load(f'{modelname}_scaler_y.bin')
        return self.model

    def save_model(self):
        """ Save model to file """
        date_now = datetime.now()
        # replace . with - in filename to look better
        timestamp_str = str(datetime.timestamp(date_now)).replace('.','-')
        self.model.save(f'roboarm_model_{timestamp_str}.h5')
        # save scalers
        dump(self.x_data_skaler, f'roboarm_model_{timestamp_str}_scaler_x.bin', compress=True)
        dump(self.y_data_skaler, f'roboarm_model_{timestamp_str}_scaler_y.bin', compress=True)


class InverseKinematics:
    """ Inverse kinematics class """
    def __init__(self, dh_matrix, joints_lengths, workspace_limits):
        self.dh_matrix = dh_matrix
        self.joints_lengths = joints_lengths
        self.workspace_limits = workspace_limits
        self.ann = ANN(self.workspace_limits, self.dh_matrix)
        # self.first_rev_joint_point = Point([0,0,dh_matrix[0]])

    def fabrik_ik(self, goal_point):
        """ Calculate angles from cosine theorem """
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

        return [theta_1, theta_2, theta_3, theta_4], [base, first_triangle, second_triangle, third_triangle]

    # use one of methods to compute inverse kinematics
    def ikine(self, method, dest_point, max_err = 0.001, max_iterations_num = 100):
        """ Calculate inverse kinematics """
        # Effector limits check
        if any(dp < limitv[1][0] or dp > limitv[1][1] for dp, limitv in zip(dest_point, self.workspace_limits.items())):
            raise Exception(f'Point {dest_point} is out of RoboArm reach area! Limits: {self.workspace_limits}')

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
            goal_joints_positions = fab.calculate(dest_point)

            # Compute roboarm angles from FABRIK computed positions
            ik_angles, _ = self.fabrik_ik(goal_joints_positions)

            return ik_angles

        elif method.lower() == "ann":
            return None

        raise Exception('Unknown method!')


class FabrikInverseKinematics(InverseKinematics):
    """ Reaching inverse kinematics using Fabrik method """

class AnnInverseKinematics(InverseKinematics):
    """ reaching inverse kinematics using Artificial NN method """
