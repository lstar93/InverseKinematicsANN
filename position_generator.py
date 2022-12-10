#!/bin/python3

import numpy as np
from math import sin, cos
from plot import *
from sklearn.preprocessing import minmax_scale

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
        return points.reshape(np.prod(points.shape[:3]), 3).tolist()

    # Cube shaped point cloud using numpy random module
    @staticmethod
    def cube_random(step, len_x, len_y, len_z):
        rr = lambda x: x*np.random.rand() # resized rand
        points = np.array([[rr(len_x), rr(len_y), rr(len_z)] for _ in np.arange(0, len_x*len_y*len_z, step)])
        return points.reshape(np.prod(points.shape[0]), 3).tolist()

    # Random Cube using python generator
    @staticmethod
    def cube_random_gen(step, len_x, len_y, len_z):
        for _ in np.arange(0, len_x*len_y*len_z, step):
            yield [len_x*np.random.rand(), len_y*np.random.rand(), len_z*np.random.rand()]

    # Random normal distribution with scaler
    @staticmethod
    def random(no_of_samples, limits=(0,1)):
        return minmax_scale(np.random.randn(no_of_samples, 3), limits)

    # Random normal distribution with scaler (generator)
    # @staticmethod
    # def random_gen(no_of_samples, limits=(0,1)):
    #    for _ in range(no_of_samples):
    #        yield minmax_scale(np.random.randn(3), limits).tolist()