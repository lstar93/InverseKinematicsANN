#!/bin/python3

# DH notation

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
  i  |  ai  |  Li  |  Ei  |  Oi  |
----------------------------------
  1  |   0  | pi/2 |  l1  |  O1  |
----------------------------------
  2  |  l2  |  0   |   0  |  O2  |
----------------------------------
  3  |  l3  |  0   |   0  |  O3  |
----------------------------------
  4  |  l4  |  0   |   0  |  O4  |
----------------------------------
Rotation matrixes:
Rt(x, L):
    [[1,         0,       0   ]
     [0,       cos(L), -sin(L)]
     [0,       sin(L),  cos(L)]]
Rt(y, B):
    [[cos(B),    0,     sin(B)]
     [0,         1,       0   ]
     [-sin(B),   0,     cos(B)]]
Rt(z, G):
    [[cos(G), -sin(G),    0   ]
     [sin(G),  cos(G),    0   ]
     [0,         0,       1   ]]
'''

'''
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
	# joints_angles_limits = {'theta_1': [-pi/2,pi/2], 'theta_2': [-pi/4,pi/2], 'theta_3': [-pi/2,pi/2], 'theta_4': [-pi/2,pi/2]} # assumed joints angles limits
	effector_workspace_limits = {'x': [1,6], 'y': [-6,6], 'z': [0,6]} # assumed limits
	links_lengths = [2, 2, 2, 2]

	# inverse kinematics engine
	ikine = InverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

	# forward kinematics
	fkine = ForwardKinematics()
	ik_angles = [] # computed joints angles

	try:
		ann = ANN(effector_workspace_limits, dh_matrix)

		no_of_samples = 1000
		positions_samples = RoboarmPositionsGenerator.random(no_of_samples, limits=(1,4))

		# calculate joints angles using FABRIK algorithm
		angles_features = [ikine.compute_roboarm_ik('FABRIK', pos) for pos in positions_samples]

		# test trajectory data
		cube_shape = [3, 3, 3]
		cube_samples_test = RoboarmPositionsGenerator.cube(0.5, *cube_shape)

		# plot training dataset
		# plot_list_points_cloud(positions_samples)
		print(len(positions_samples))
		print(np.array(positions_samples).shape)

		# train model using generated dataset
		epochs=2000
		ann.train_model(epochs, positions_samples, angles_features) # random data

		# use existing model
		# ann.load_model('roboarm_model_1664488076-610064.h5')

		# print/plot test points dataset
		# test_points = [Point([*elem]) for elem in cube_samples_test]
		# plot_points_cloud(test_points)
		# print('sample: ' + str(np.array(cube_samples_test)))

		# predict positions on generated data
		predicted_points = []
		ik_angles_ann = ann.predict_ik(cube_samples_test).tolist()

		# compute FK to check ANN IK
		for angles in ik_angles_ann:
			fk, _ = fkine.forward_kinematics(*[angles, dh_matrix[1:]])
			predicted_points.append(fk.T)

		# print/plot predicted points
		# plot_points_cloud(predicted_points)
		print(len(predicted_points))
		print(np.array(predicted_points).shape)
		# print('predicted: ' + str(np.array(predicted_points)))

		print(np.array(cube_samples_test) - np.array(predicted_points))

		# save exceptional models
		# ann.save_model()

	except Exception as e:
		print(str(e))