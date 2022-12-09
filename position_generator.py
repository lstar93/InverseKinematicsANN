#!/bin/python3

import random as rand
import numpy as np
from math import sin, cos
from scipy.stats import truncnorm

# Generate learning data for ANN
class RoboarmPositionsGenerator:
    
    # Circle
    @staticmethod
    def circle(radius, no_of_samples, centre):
        pos = lambda t: [centre[0], centre[1] * sin(t), centre[2] + radius*cos(t)]
        return np.array([pos(t) for t in range(no_of_samples)])

    # Circle using generator
    @staticmethod
    def circle_gen(radius, no_of_samples, centre):
        for t in range(no_of_samples):
            yield np.array(centre[0], centre[1] * sin(t), centre[2] + radius*cos(t))
        
    # Cube
    @staticmethod
    def cube(step_size, len_x, len_y, len_z):
        positions = []
        for z in range(len_z):
            for y in range(len_y):
                for x in range(len_x):
                    positions.append([x*step_size, y*step_size, z*step_size])
        return np.array(positions)

    @staticmethod
    def cube_random(len_x, len_y, len_z):
        matrix_3d = np.random(len_x*len_y*len_x).reshape(len_x, len_y, len_z)
        return matrix_3d

    @staticmethod
    def cube_gen(step_size, len_x, len_y, len_z):
        for x in range(len_x):
            yield [x*step_size, y*step_size, z*step_size]
        for y in range(len_y):
            yield [x*step_size, y*step_size, z*step_size]
        for z in range(len_z):
            yield [x*step_size, y*step_size, z*step_size]

    # Random distribution
    @staticmethod
    def random(no_of_samples, limits, distribution = 'normal'):
        positions = [] # output samples -> angles
        for _, limitv in limits.items():
            if distribution == 'normal':
                mean = 0
                sd = 0.5
                positions.append(list(truncnorm((limitv[0] - mean) / 0.5, (limitv[1] - mean) / sd, loc=mean, scale=sd)))
            elif distribution == 'uniform':
                positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            elif distribution == 'random':
                # just random shuffled data
                arr = np.linspace(limitv[0],limitv[1],no_of_samples)
                np.random.shuffle(arr)
                positions.append(arr)
            else:
                raise Exception('Unknown distribution, use: \'normal\' (default), \'unifrom\', \'random\'')
        return np.array(positions).T