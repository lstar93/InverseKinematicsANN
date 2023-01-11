#!/bin/python3

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

from position_generator import *
from forward_kinematics import *
from inverse_kinematics import *
from plot import *
import sys

if __name__ == '__main__':

	# 6 DOF robot DH matrix
	dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]

	# links lengths, workspace and joints limits
	effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [0,6]} # assumed limits
	links_lengths = [2, 2, 2, 2]

	# inverse kinematics engine
	ikine = InverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

	# forward kinematics
	fkine = ForwardKinematics()

	#try:
	### CREATE MODEL
	ann = ANN(effector_workspace_limits, dh_matrix)

	# prepare training data
	# no_of_samples = 10
	# positions_samples = RoboarmTrainingDataGenerator.cube_random(0.5, 5, 12, 6, (1,-6,0))
	# print(len(positions_samples))
	# print(np.array(positions_samples).shape)
	# positions_samples = RoboarmTrainingDataGenerator.random(20000, limits=effector_workspace_limits)
	# positions_samples = RoboarmTrainingDataGenerator.random_distribution(no_of_samples = 50000, limits = effector_workspace_limits, distribution='random')
	# angles_features = [ikine.ikine('FABRIK', pos) for pos in positions_samples] # use FABRIK to prepare train/test features

	# train model using generated dataset
	# epochs = 1000
	# ann.train_model(epochs, positions_samples, angles_features) # random data
	# ann.fit_trainig_data(positions_samples, angles_features)

	# use existing model
	ann.load_model('roboarm_model_1672332006-446743.h5')

	### TEST MODEL

	# test trajectory data
	test_shape = [3, 3, 3]
	test_samples_test = RoboarmTrainingDataGenerator.cube(1, *test_shape)
	plot_points_3d(test_samples_test)

	# test trajectory using circle
	# radius = 5
	# no_of_samples = 100
	# centre = [1,3,1]
	# test_samples_test = RoboarmTrainingDataGenerator.circle(radius, no_of_samples, centre)
	# plot_points_3d(test_samples_test)

	# test_samples_test = RoboarmTrainingDataGenerator.random(5, limits=effector_workspace_limits)
	# plot_points_3d(test_samples_test)
	# print(list(test_samples_test))

	# predict positions on generated data
	predicted_points = []
	ik_angles_ann = ann.predict_ik(test_samples_test).tolist()

	# compute FK to check ANN IK
	for angles in ik_angles_ann:
		dh_matrix_out = [angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
		fk, _ = fkine.fkine(*dh_matrix_out)
		predicted_points.append([fk[0,3], fk[1,3], fk[2,3]])

	# print/plot predicted points
	plot_points_3d(predicted_points, path=False)
	print(list(predicted_points))

	# print(len(predicted_points))
	# print(np.array(predicted_points).shape)
	# print('predicted: ' + str(predicted_points))
	# print(np.array(cube_samples_test) - np.array(predicted_points))

	# save exceptional models
	# ann.save_model()

	# except Exception as e:
	# 	print(str(e))
