""" Plotters for robot environment """
#!/usr/bin/env python

# pylint: disable=W0511 # suppress TODOs

from math import sin, cos
import random as rand
import numpy as np
from scipy.stats import truncnorm

from sklearn.preprocessing import minmax_scale
import keras


def transpose(data):
    """ Transpose list """
    return list(map(list, zip(*data)))

def get_truncated_normal_distribution(mean=0, std_dev=1, low=0, upp=10):
    """ Truncated normal distribution """
    return truncnorm((low - mean) / std_dev, (upp - mean) / std_dev, loc=mean, scale=std_dev)


class RoboarmTrainingDataGenerator:
    """ Class used to enerate learn data for ANN """

    @staticmethod
    def circle(radius, no_of_samples, centre):
        """ Circle shape """
        def generate_shape(tstamp):
            return [centre[0], centre[1] * sin(tstamp), centre[2] + radius*cos(tstamp)]
        return [generate_shape(t) for t in range(no_of_samples)]

    @staticmethod
    def circle_gen(radius, no_of_samples, centre):
        """ Circle shape generator """
        for tstamp in range(no_of_samples):
            yield [centre[0], centre[1] * sin(tstamp), centre[2] + radius*cos(tstamp)]

    @staticmethod
    def cube(step, len_x, len_y, len_z, start=(0,0,0)):
        """ Cube shape """
        points = np.array([[[ [x+start[0], y+start[1], z+start[2]]
                                for x in np.arange(0, len_x, step) ]
                                    for y in np.arange(0, len_y, step) ]
                                        for z in np.arange(0, len_z, step)])
        return points.reshape(np.prod(points.shape[:3]), 3).tolist()

    @staticmethod
    def cube_random(step, len_x, len_y, len_z, start=(0,0,0)):
        """ Cube shaped point cloud using numpy random module """
        def resizer(axis): # resized rand
            return axis*np.random.rand()
        pts = np.array([[resizer(len_x)+start[0], resizer(len_y)+start[1], resizer(len_z)+start[2]]
                            for _ in np.arange(0, len_x*len_y*len_z, step)])
        return pts.reshape(np.prod(pts.shape[0]), 3).tolist()

    @staticmethod
    def cube_random_gen(step, len_x, len_y, len_z, start=(0,0,0)):
        """ Random Cube using python generator """
        for _ in np.arange(0, len_x*len_y*len_z, step):
            yield [len_x*np.random.rand()+start[0],
                   len_y*np.random.rand()+start[1],
                   len_z*np.random.rand()+start[2]]

    @staticmethod
    def random(no_of_samples, limits):
        """ Random normal distribution with scaler """
        def apply_limits(axis):
            return minmax_scale(np.random.randn(no_of_samples), limits[axis])
        return [[x,y,z] for x,y,z in zip(apply_limits('x'), apply_limits('y'), apply_limits('z'))]

    @staticmethod
    def spring(no_of_samples, len_x, len_y, len_z):
        """ Horizontal spring shape """
        axis_z = np.linspace(0, len_z, no_of_samples)
        axis_x = ((np.sin(axis_z)*len_x)+len_x)
        axis_y = ((np.cos(axis_z)*len_y)+len_y)
        return [[x/2,y/2,z] for x,y,z in zip(axis_x, axis_y, axis_z)]

    # Random trajectory
    @staticmethod
    def random_distribution(no_of_samples, limits, distribution = 'normal', std_dev=0.5):
        """ Different ways to create randomly generated data """
        positions = [] # output samples -> angles
        for _, limitv in limits.items():
            if distribution == 'normal':
                val = list(get_truncated_normal_distribution(
                            mean=0, std_dev=std_dev,
                            low=limitv[0], upp=limitv[1])
                            .rvs(no_of_samples))
                positions.append(val)
            elif distribution == 'uniform':
                positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            elif distribution == 'random': # just randomly shuffled data
                arr = np.linspace(limitv[0], limitv[1], no_of_samples)
                np.random.shuffle(arr)
                positions.append(arr)
        return transpose(positions)

# TODO: check generator unexpected exchaustion
class CubeDataGenerator(keras.utils.Sequence):
    """ Cube generator class for keras fit method """
    def __init__(self, ikine, generator, datalen, batch_size):
        self.ikine = ikine
        self.generator = generator
        self.datalen = datalen
        self.batch_size = batch_size

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        x_batch = np.array([next(self.generator) for _ in range(0, self.batch_size)])
        # use FABRIK inverse kinematics to prepare train/test features
        y_batch = np.array([self.ikine.ikine('FABRIK', pos) for pos in x_batch])
        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size
