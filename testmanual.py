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
1. read robot configuration (with angles limits) from file
'''

import sys
from kinematics.point import Point
from kinematics.forward import ForwardKinematics
from kinematics.inverse import FabrikInverseKinematics, AnnInverseKinematics
from plot.plot import Plotter
from robot.position_generator import TrainingDataGenerator
from robot.robot import SixDOFRobot as Robot
from robot.robot import OutOfRobotReachException

if __name__ == '__main__':

    # inverse kinematics engine
    ikine_fab = FabrikInverseKinematics(Robot.dh_matrix, Robot.links_lengths, Robot.effector_workspace_limits)

    ikine_ann = AnnInverseKinematics(Robot.dh_matrix, Robot.links_lengths, Robot.effector_workspace_limits)
    ikine_ann.load_model('models/roboarm_model_1674153800-982793.h5')

    # forward kinematics
    fkine = ForwardKinematics(Robot.dh_matrix)

    '''
    #try:
    ### CREATE MODEL
    ann = ANN(effector_workspace_limits, dh_matrix)

    positions_samples_0 = TrainingDataGenerator.cube_random(0.0033, 5, 12, 6, (1,-6,-2))

    positions_samples_1 = TrainingDataGenerator.cube_random(0.0033, 6, 12, 2, (0,-6,4))

    positions_samples_2 = TrainingDataGenerator.random(20000, limits=effector_workspace_limits)

    positions_samples_3 = TrainingDataGenerator.random_distribution(
														no_of_samples = 20000,
														limits = effector_workspace_limits,
														distribution='normal',
														std_dev=0.33)

    positions_samples_4 = TrainingDataGenerator.cube_random(0.0033, 2, 12, 4, (0,-6,-2))

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
    # gen = CubeDataGenerator(ikine, TrainingDataGenerator.cube_random_gen(0.01, 5, 12, 6, (1,-6,0)), 15000, 64)
    # ann.train_model(epochs=1000, features=[], samples=[], generator=gen) # random data
    positions_samples = TrainingDataGenerator.cube(0.5, 2, 2, 2, (0.5, 0.5, 0.5))
    
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
    ann.load_model('roboarm_model_1674153800-982793.h5')

    ### TEST MODEL

    # test trajectory data
    test_shape = [2, 2, 2]
    test_sample = TrainingDataGenerator.cube(0.5, *test_shape, start=(1,0,1))
    plot_points_3d(test_sample)
    predicted = predict(test_sample, True, True)
    plot_joint_points_3d(predicted, test_sample)

    # test trajectory using circle
    radius = 2
    no_of_samples = 30
    centre = [1,3,1]
    test_sample = TrainingDataGenerator.circle(radius, no_of_samples, centre)
    plot_points_3d(test_sample)
    predicted = predict(test_sample, True, True)
    plot_joint_points_3d(predicted, test_sample)

    spring_size = [2, 2, 4]
    test_sample = TrainingDataGenerator.spring(no_of_samples, *spring_size)
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

    # points = TrainingDataGenerator.cube(0.4, 2, 2, 4, (-1, -1, -2))
    # plot_points_3d(points)
    points = TrainingDataGenerator.spring(50, 4, 4, 4)
    # points = TrainingDataGenerator.circle(2, 50, (1,1,1))

    predicted_points = []
    # compute FK to check ANN IK
    for point in points:
        predicted_all_joints_position = [Point([0, 0, 0])]
        try:
            angles = ikine_fab.ikine([point])
            fk, fkall = fkine.fkine(angles[0])
            predicted_point = [fk[0,3], fk[1,3], fk[2,3]]
            predicted_points.append(predicted_point)
            predicted_all_joints_position += \
                [Point([fka[0,3], fka[1,3], fka[2,3]]) for fka in fkall]
        except OutOfRobotReachException as exception:
            print(str(exception))
            sys.exit(0)

        Plotter.plot_robot(predicted_all_joints_position, points)

    # print(predicted_points)
    # Plotter.set_limits((0.998,1.002), (-1,3), (-1,3))
    # Plotter.plot_points_3d(points, dot_color='r')
    # Plotter.plot_points_3d(predicted_points, dot_color='b')
    # Plotter.plot_joint_points_3d(points, predicted_points)
