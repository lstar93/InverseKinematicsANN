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
	effector_workspace_limits = {'x': [1,6], 'y': [-6,6], 'z': [0,6]} # assumed limits
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
	positions_samples = RoboarmTrainingDataGenerator.cube_random(0.5, 5, 12, 6, (1,-6,0))
	print(len(positions_samples))
	print(np.array(positions_samples).shape)

	# calculate joints angles using FABRIK algorithm
	angles_features = [ikine.ikine('FABRIK', pos) for pos in positions_samples]

	# train model using generated dataset
	# ann.train_model(epochs=2000, positions_samples, angles_features) # random data
	# ann.fit_trainig_data(positions_samples, angles_features)
	# use existing model
	ann.load_model('roboarm_model_1664488076-610064.h5')

	### TEST MODEL

	# test trajectory data
	# test_shape = [3, 3, 3]
	# test_samples_test = RoboarmTrainingDataGenerator.random(50)
	# plot_list_points_cloud(test_samples_test)

	# test trajectory using circle
	radius = 5
	no_of_samples = 100
	centre = [1,3,1]
	test_samples_test = RoboarmTrainingDataGenerator.circle(radius, no_of_samples, centre)
	plot_list_points_cloud(test_samples_test)

	# predict positions on generated data
	predicted_points = []
	ik_angles_ann = ann.predict_ik(test_samples_test).tolist()

	# compute FK to check ANN IK
	for angles in ik_angles_ann:
		fk, _ = fkine.fkine(*[angles, *dh_matrix[1:]])
		predicted_points.append([fk[0,3], fk[1,3], fk[2,3]])

	# print/plot predicted points
	plot_list_points_cloud(predicted_points)
	# print(len(predicted_points))
	# print(np.array(predicted_points).shape)
	# print('predicted: ' + str(predicted_points))
	# print(np.array(cube_samples_test) - np.array(predicted_points))

	# save exceptional models
	# ann.save_model()

	# except Exception as e:
	# 	print(str(e))
