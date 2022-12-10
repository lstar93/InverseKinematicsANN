#!/bin/python3

import random as rand
import numpy as np
from math import sin, cos
from scipy.stats import truncnorm
from plot import *

# Generate learn data for ANN
class RoboarmPositionsGenerator:
    
    # Circle
    @staticmethod
    def circle(radius, no_of_samples, centre):
        pos = lambda t: [centre[0], centre[1] * sin(t), centre[2] + radius*cos(t)]
        return [pos(t) for t in range(no_of_samples)]

    # Circle using generator
    @staticmethod
    def circle_gen(radius, no_of_samples, centre):
        for t in range(no_of_samples):
            yield [centre[0], centre[1] * sin(t), centre[2] + radius*cos(t)]

    # Cube
    @staticmethod
    def cube(step, len_x, len_y, len_z):
        points = np.array([[[ [x,y,z] for x in np.arange(0, len_x, step) ] for y in np.arange(0, len_y, step) ] for z in np.arange(0, len_z, step)])
        return points.reshape(np.prod(points.shape[:3]), 3)

    # Cube shaped point cloud using numpy random module
    @staticmethod
    def cube_random(step, len_x, len_y, len_z):
        rr = lambda x: x*np.random.rand() # resized rand
        points = np.array([[rr(len_x), rr(len_y), rr(len_z)] for _ in np.arange(0, len_x*len_y*len_z, step)])
        return points.reshape(np.prod(points.shape[0]), 3)

    # Random Cube using python generator
    @staticmethod
    def cube_random_gen(step, len_x, len_y, len_z):
        for _ in np.arange(0, len_x*len_y*len_z, step):
            yield [len_x*np.random.rand(), len_y*np.random.rand(), len_z*np.random.rand()]

    # Random distribution
    @staticmethod
    def random(no_of_samples, limits, distribution = 'normal'):
        # positions = [] # output samples -> angles
        # for _, limitv in limits.items():
        if distribution == 'normal':
            mean = 0
            sd = 0.5
            return np.array([[truncnorm((limitv[0] - mean) / 0.5, (limitv[1] - mean) / sd, loc=mean, scale=sd)] for _, limitv in limits.items()]).T.tolist()
        elif distribution == 'uniform':
            # positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            return np.array([[rand.uniform(*limitv) for x in range(no_of_samples)] for _, limitv in limits.items()]).T.tolist()
        elif distribution == 'random':
            # just random shuffled data
            # np.random.shuffle(np.linspace(limitv[0],limitv[1],no_of_samples))
            # positions.append(arr)
            return np.array([np.random.shuffle(np.linspace(limitv[0],limitv[1],no_of_samples)) for _, limitv in limits.items()]).T.tolist()
        else:
            raise Exception('Unknown distribution, use: \'normal\' (default), \'unifrom\', \'random\'')

# circle
radius = 5
no_of_samples = 20
centre = [1,3,1]

# cube
step = 1
cube = [5, 5, 5]

# TODO: check round

# OK
# circle = RoboarmPositionsGenerator.circle(radius, no_of_samples, centre)
# plot_list_points_cloud(circle)

# OK
# circle_generator = RoboarmPositionsGenerator.circle_gen(radius, no_of_samples, centre)
# circleg = [next(circle_generator) for x in range(no_of_samples)]
# plot_list_points_cloud(circleg)

# OK
# cube = RoboarmPositionsGenerator.cube(step, *cube)
# print(cube)
# plot_list_points_cloud(cube)

# OK
# cube_random = RoboarmPositionsGenerator.cube_random(step, *cube)
# print(cube_random)
# plot_list_points_cloud(cube_random)

# OK
# cube_gen = RoboarmPositionsGenerator.cube_random_gen(step, *cube)
# cubeg = [next(cube_gen) for _ in np.arange(0, np.prod(cube), step)]
# print(cubeg)
# plot_list_points_cloud(cubeg)