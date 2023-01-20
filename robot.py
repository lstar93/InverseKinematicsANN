#!/bin/python3

# pylint: disable=W0511 # suppress TODOs
# pylint: disable=W1401 # string constant might be missing r prefix
# pylint: disable=W0105 # string has no effect

'''
      2y |          | 3y
         |     l3   |
         0-->-------0--> 3x
        /  2x        \
   y1| / l2        l4 \ |4y
     |/                \|
  1z 0-->1x          4z 0-->4x
     |                 ----
     | l1              |  |
    /_\
    \ /
     |
_____|_____

TODO:
1. read robot configuration (with angle limits) from file
2. simple CLI to choose/save model and data generator
3. add CSV positions handling via CLI
'''

from math import pi
import sys
import numpy as np
from forward_kinematics import ForwardKinematics
from inverse_kinematics import ANN, FabrikInverseKinematics, AnnInverseKinematics
from plot import plot_points_3d, plot_joint_points_3d
from position_generator import RoboarmTrainingDataGenerator

def predict(test_samples_test, separate_predictions=False, plot=False):
    """ Predict thetas using neural network """
    predicted_points = []
    ik_angles_ann = []

    if separate_predictions:
        for sample in test_samples_test:
            ik_angles_ann.append(ann.predict([sample]).tolist()[0])
    else:
        ik_angles_ann = ann.predict(test_samples_test).tolist()

    # compute FK to check ANN IK
    for angles in ik_angles_ann:
        dh_matrix_out = [angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
        fk, _ = fkine.fkine(*dh_matrix_out)
        predicted_points.append([fk[0,3], fk[1,3], fk[2,3]])

    # print/plot predicted points
    if plot:
        plot_points_3d(predicted_points, path=False)
    return predicted_points

if __name__ == '__main__':

    # 6 DOF robot DH matrix
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]

    # links lengths, workspace and joints limits
    effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]} # assumed limits
    links_lengths = [2, 2, 2, 2]

    # inverse kinematics engine
    ikine = FabrikInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

    ikine_ann = AnnInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)
    ikine_ann.load_model('roboarm_model_1674153800-982793.h5')

    # forward kinematics
    fkine = ForwardKinematics()

    #try:
    ### CREATE MODEL
    ann = ANN(effector_workspace_limits, dh_matrix)

    '''
    positions_samples_0 = RoboarmTrainingDataGenerator.cube_random(0.0033, 5, 12, 6, (1,-6,-2))

    positions_samples_1 = RoboarmTrainingDataGenerator.cube_random(0.0033, 6, 12, 2, (0,-6,4))

    positions_samples_2 = RoboarmTrainingDataGenerator.random(20000, limits=effector_workspace_limits)

    positions_samples_3 = RoboarmTrainingDataGenerator.random_distribution(
														no_of_samples = 20000,
														limits = effector_workspace_limits,
														distribution='normal',
														std_dev=0.33)

    positions_samples_4 = RoboarmTrainingDataGenerator.cube_random(0.0033, 2, 12, 4, (0,-6,-2))

    # plot_points_3d(positions_samples)

    positions_samples = []
    for ps0, ps1, ps2, ps3, ps4 in zip(positions_samples_0, positions_samples_1, positions_samples_2, positions_samples_3, positions_samples_4):
        positions_samples.append(ps0)
        positions_samples.append(ps1)
        positions_samples.append(ps2)
        positions_samples.append(ps3)
        positions_samples.append(ps4)
    # np.random.shuffle(positions_samples)

    for pos in positions_samples:
        for i, elem in enumerate(pos):
            pos[i] = round(elem, 10)
            if pos[i] > 6:
                print(pos[i])
                sys.exit(0)
    angles_features = [ikine.ikine('FABRIK', pos) for pos in positions_samples] # use FABRIK to prepare train/test features

    # train model using generated dataset
    epochs = 1000
    ann.train_model(epochs, positions_samples, angles_features) # random data
    # gen = CubeDataGenerator(ikine, RoboarmTrainingDataGenerator.cube_random_gen(0.01, 5, 12, 6, (1,-6,0)), 15000, 64)
    # ann.train_model(epochs=1000, features=[], samples=[], generator=gen) # random data
    '''
    positions_samples = RoboarmTrainingDataGenerator.cube(0.5, 2, 2, 2, (0.5, 0.5, 0.5))
    
    angles_features = [ikine.ikine(pos) for pos in positions_samples] # use FABRIK to prepare train/test features
    # angles_features = [ikine_ann.ikine([pos]) for pos in positions_samples]
    # angles_features = ikine_ann.ikine(positions_samples, True)

    shape = []
    for angles in angles_features:
        dh_matrix_out = [angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
        fk, _ = fkine.fkine(*dh_matrix_out)
        shape.append([fk[0,3], fk[1,3], fk[2,3]])

    plot_joint_points_3d(shape, positions_samples)

    # use existing model
    '''
    ann.load_model('roboarm_model_1674153800-982793.h5')

    ### TEST MODEL

    # test trajectory data
    test_shape = [2, 2, 2]
    test_sample = RoboarmTrainingDataGenerator.cube(0.5, *test_shape, start=(1,0,1))
    plot_points_3d(test_sample)
    predicted = predict(test_sample, True, True)
    plot_joint_points_3d(predicted, test_sample)

    # test trajectory using circle
    radius = 2
    no_of_samples = 30
    centre = [1,3,1]
    test_sample = RoboarmTrainingDataGenerator.circle(radius, no_of_samples, centre)
    plot_points_3d(test_sample)
    predicted = predict(test_sample, True, True)
    plot_joint_points_3d(predicted, test_sample)

    spring_size = [2, 2, 4]
    test_sample = RoboarmTrainingDataGenerator.spring(no_of_samples, *spring_size)
    plot_points_3d(test_sample)
    predicted = predict(test_sample, True, True)
    plot_joint_points_3d(predicted, test_sample, True)

    # save exceptional models
    response = input('Save this model? [y/n]:')
    if response == 'y':
        ann.save_model()

    # except Exception as e:
    #     print(str(e))
    '''
