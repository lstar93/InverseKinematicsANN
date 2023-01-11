#!/bin/python3

import numpy as np
from math import sin, cos
from plot import *
from sklearn.preprocessing import minmax_scale

from scipy.stats import truncnorm
import random as rand

def transpose(data):
    return list(map(list, zip(*data))) # transpose [[x,y,z], ...] into columns [[x,...], [y,...], [z,...]] 

def get_truncated_normal_distribution(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# Generate learn data for ANN
class RoboarmTrainingDataGenerator:
    
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
    def cube(step, len_x, len_y, len_z, start=(0,0,0)):
        points = np.array([[[ [x+start[0], y+start[1], z+start[2]] 
                                for x in np.arange(0, len_x, step) ] 
                                    for y in np.arange(0, len_y, step) ] 
                                        for z in np.arange(0, len_z, step)])
        return points.reshape(np.prod(points.shape[:3]), 3).tolist()

    # Cube shaped point cloud using numpy random module
    @staticmethod
    def cube_random(step, len_x, len_y, len_z, start=(0,0,0)):
        rr = lambda x: x*np.random.rand() # resized rand
        points = np.array([[rr(len_x)+start[0], rr(len_y)+start[1], rr(len_z)+start[2]] for _ in np.arange(0, len_x*len_y*len_z, step)])
        return points.reshape(np.prod(points.shape[0]), 3).tolist()

    # Random Cube using python generator
    @staticmethod
    def cube_random_gen(step, len_x, len_y, len_z, start=(0,0,0)):
        for _ in np.arange(0, len_x*len_y*len_z, step):
            yield [len_x*np.random.rand()+start[0], len_y*np.random.rand()+start[1], len_z*np.random.rand()+start[2]]

    # Random normal distribution with scaler
    @staticmethod
    def random(no_of_samples, limits={'x':[0,1], 'y':[0,1], 'z':[0,1]}):
        pts = lambda axis: minmax_scale(np.random.randn(no_of_samples), limits[axis])
        return [[x,y,z] for x,y,z in zip(pts('x'), pts('y'), pts('z'))]

    # Random normal distribution with scaler (generator)
    # @staticmethod
    # def random_gen(no_of_samples, limits=(0,1)):
    #    for _ in range(no_of_samples):
    #        yield minmax_scale(np.random.randn(3), limits).tolist()

    # Random trajectory
    @staticmethod
    def random_distribution(no_of_samples, limits, distribution = 'normal'):
        positions = [] # output samples -> angles
        for _, limitv in limits.items():
            if distribution == 'normal':
                val = list(get_truncated_normal_distribution(mean=0, sd=0.5, low=limitv[0], upp=limitv[1]).rvs(no_of_samples))
                positions.append(val)
            elif distribution == 'uniform':
                positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            elif distribution == 'random':
                # just random shuffled data
                arr = np.linspace(limitv[0], limitv[1], no_of_samples)
                np.random.shuffle(arr)
                positions.append(arr)
        return transpose(positions)