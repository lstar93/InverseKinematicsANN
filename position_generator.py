#!/bin/python3

from math import sin, cos
import random as rand
import numpy as np
from scipy.stats import truncnorm

from sklearn.preprocessing import minmax_scale
import keras


def transpose(data):
    """ Transpose list """
    return list(map(list, zip(*data)))


def get_truncated_normal_distribution(mean=0, sd=1, low=0, upp=10):
    """ Truncated normal distribution """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class RoboarmTrainingDataGenerator:
    """ Class used to enerate learn data for ANN """

    @staticmethod
    def circle(radius, no_of_samples, centre):
        """ Circle shape """
        pos = lambda t: [centre[0], centre[1] * sin(t), centre[2] + radius*cos(t)]
        return [pos(t) for t in range(no_of_samples)]

    @staticmethod
    def circle_gen(radius, no_of_samples, centre):
        """ Circle shape generator """
        for t in range(no_of_samples):
            yield [centre[0], centre[1] * sin(t), centre[2] + radius*cos(t)]

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
        rr = lambda x: x*np.random.rand() # resized rand
        points = np.array([[rr(len_x)+start[0], rr(len_y)+start[1], rr(len_z)+start[2]] for _ in np.arange(0, len_x*len_y*len_z, step)])
        return points.reshape(np.prod(points.shape[0]), 3).tolist()

    @staticmethod
    def cube_random_gen(step, len_x, len_y, len_z, start=(0,0,0)):
        """ Random Cube using python generator """
        for _ in np.arange(0, len_x*len_y*len_z, step):
            yield [len_x*np.random.rand()+start[0], len_y*np.random.rand()+start[1], len_z*np.random.rand()+start[2]]

    @staticmethod
    def random(no_of_samples, limits={'x':[0,1], 'y':[0,1], 'z':[0,1]}):
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

    # Random normal distribution with scaler (generator)
    # @staticmethod
    # def random_gen(no_of_samples, limits=(0,1)):
    #    for _ in range(no_of_samples):
    #        yield minmax_scale(np.random.randn(3), limits).tolist()

    # Random trajectory
    @staticmethod
    def random_distribution(no_of_samples, limits, distribution = 'normal', sd=0.5):
        positions = [] # output samples -> angles
        for _, limitv in limits.items():
            if distribution == 'normal':
                val = list(get_truncated_normal_distribution(mean=0, sd=sd, low=limitv[0], upp=limitv[1]).rvs(no_of_samples))
                positions.append(val)
            elif distribution == 'uniform':
                positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            elif distribution == 'random':
                # just random shuffled data
                arr = np.linspace(limitv[0], limitv[1], no_of_samples)
                np.random.shuffle(arr)
                positions.append(arr)
        return transpose(positions)


class CubeDataGenerator(keras.utils.Sequence):
    """ Cube generator class for keras fit method """

    def __init__(self, ikine, limits, shape, step, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.ikine = ikine
        self.limits = limits
        self.shuffle = shuffle
        self.shape = shape
        self.step = step
        self.generator = RoboarmTrainingDataGenerator.cube_random_gen(self.step, *self.shape)
        self.datalen = len(np.arange(0, shape[0]*shape[1]*shape[2], step))

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        x_batch = [next(self.generator) for _ in range(0,self.batch_size)]
        # use FABRIK to prepare train/test features
        y_batch = [self.ikine.ikine('FABRIK', pos) for pos in x_batch]

        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size


class SequenceExample(keras.utils.Sequence):

    def __init__(self, x_in, y_in, batch_size, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
