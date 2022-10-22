
#!/bin/python3

import random as rand
import numpy as np
from math import sin, cos
from scipy.stats import truncnorm

# Generate learning data for ANN
class RoboarmPositionsGenerator:
    
    def transpose(self, data):
        return list(map(list, zip(*data))) # transpose [[x,y,z], ...] into columns [[x,...], [y,...], [z,...]] 

    # Circle
    def circle(self, radius, no_of_samples, position):
        positions=[]
        for t in range(no_of_samples):
            x=position[0]
            y=position[1] * sin(t)
            z=position[2] + radius*cos(t)
            positions.append([x, y, z])
        # if verbose:
        #     plot_points_3d([Point([*elem]) for elem in positions])
        return positions

    # Cube
    def cube(self, step_size, limits):
        positions=[]
        all_x = []
        all_y = []
        all_z = []
        for x in np.linspace(*limits['x_limits'], step_size):
            for y in np.linspace(*limits['y_limits'], step_size):
                all_x += list(np.linspace(x,x,step_size))
                all_y += list(np.linspace(y,y,step_size))
                all_z += list(np.linspace(*limits['z_limits'], step_size))
        for x, y, z in zip(all_x, all_y, all_z):
            positions.append([x,y,z])
        # if verbose:
        #     plot_points_3d([Point([*elem]) for elem in positions])
        return positions

    def get_truncated_normal_distribution(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    # Random distribution
    def random(self, no_of_samples, limits, distribution = 'normal'):
        positions = [] # output samples -> angles
        for _, limitv in limits.items():
            if distribution == 'normal':
                val = list(self.get_truncated_normal_distribution(mean=0, sd=0.5, low=limitv[0], upp=limitv[1]).rvs(no_of_samples))
                positions.append(val)
            elif distribution == 'uniform':
                positions.append([rand.uniform(*limitv) for x in range(no_of_samples)])
            elif distribution == 'random':
                # just random shuffled data
                arr = np.linspace(limitv[0],limitv[1],no_of_samples)
                np.random.shuffle(arr)
                positions.append(arr)
            else:
                raise Exception('Unknown distribution, use: \'normal\' (default), \'unifrom\', \'random\'')
        # if verbose:
        #     plot_points_3d([Point([*elem]) for elem in self.transpose(positions)])
        return self.transpose(positions)