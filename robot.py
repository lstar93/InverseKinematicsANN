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
from position_generator import CubeDataGenerator
import sys

def predict(ann, fkine, test_samples_test, separate_predictions=False, plot=False):
	predicted_points = []
	ik_angles_ann = []

	if separate_predictions:
		for sample in test_samples_test:
			ik_angles_ann.append(ann.predict_ik([sample]).tolist()[0])
	else:
		ik_angles_ann = ann.predict_ik(test_samples_test).tolist()

	# compute FK to check ANN IK
	for angles in ik_angles_ann:
		dh_matrix_out = [angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
		fk, _ = fkine.fkine(*dh_matrix_out)
		predicted_points.append([fk[0,3], fk[1,3], fk[2,3]])

	# print/plot predicted points
	if(plot):
		plot_points_3d(predicted_points, path=False)
	return predicted_points

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
	positions_samples_0 = RoboarmTrainingDataGenerator.cube_random(0.01, 5, 12, 6, (1,-6,0))
	# print(len(positions_samples))
	# print(np.array(positions_samples).shape)
	positions_samples_1 = RoboarmTrainingDataGenerator.random(15000, limits=effector_workspace_limits)
	# plot_points_3d(positions_samples)

	positions_samples_2 = RoboarmTrainingDataGenerator.random_distribution(no_of_samples = 15000, limits = effector_workspace_limits, distribution='normal', sd=0.33)
	# plot_points_3d(positions_samples)

	positions_samples = []
	for first, sec, thrd in zip(positions_samples_0, positions_samples_1, positions_samples_2):
		positions_samples.append(first)
		positions_samples.append(sec)
		positions_samples.append(thrd)

	np.random.shuffle(positions_samples)

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
	# ann.train_model(epochs=epochs, samples=[], features=[], generator=CubeDataGenerator(ikine=ikine, limits=effector_workspace_limits, shape=[5,5,5], step=0.1, batch_size=64)) # random data

	# use existing model
	# ann.load_model('roboarm_model_1673989748-812473.h5')

	### TEST MODEL

	# test trajectory data
	test_shape = [2, 2, 2]
	test_sample = RoboarmTrainingDataGenerator.cube(0.5, *test_shape, start=(1,0,1))
	plot_points_3d(test_sample)
	predicted = predict(ann, fkine, test_sample, True, True)
	print(test_sample[0:10])
	print(predicted[0:10])
	print(np.array(test_sample[0:10]) - np.array(predicted[0:10]))

	# test trajectory using circle
	radius = 2
	no_of_samples = 30
	centre = [1,3,1]
	test_sample = RoboarmTrainingDataGenerator.circle(radius, no_of_samples, centre)
	plot_points_3d(test_sample)
	predicted = predict(ann, fkine, test_sample, True, True)
	print(test_sample[0:10])
	print(predicted[0:10])
	print(np.array(test_sample[0:10]) - np.array(predicted[0:10]))

	spring_size = [2, 2, 4]
	test_sample = RoboarmTrainingDataGenerator.spring(no_of_samples, *spring_size)
	plot_points_3d(test_sample)
	predicted = predict(ann, fkine, test_sample, True, True)

	# save exceptional models
	# ann.save_model()

	# except Exception as e:
	# 	print(str(e))
