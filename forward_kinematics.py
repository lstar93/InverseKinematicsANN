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

import numpy as np
from math import sin, cos, pi

# supress printing enormous small numbers like 0.123e-16
np.set_printoptions(suppress=True)

class ForwardKinematics:
    # Rotation matrix
    # rot_axis -> rotation axis -> 'x', 'y' or 'z'
    # angle -> rotation angle in radians
    # size -> dimention of square matrix, defualt and minimum is 3

    # robot_features_size -> number of robot features determining shape of 
    # transforamtion, translation and rotation matrixes
    def __init__(self, robot_features_size = 0) -> None:
        self.robot_features_size = robot_features_size

    # generate rotation matrix via rotation axis
    def rotation_matrix(self, rot_axis, angle):
        if (angle < -2*pi) or (angle > 2*pi):
            raise Exception('Error, angle limits are from -2pi to 2pi')
        if self.robot_features_size < 3:
            raise Exception('Error, rotation matrix size and shape should be equal to 3 or greater')

        # Generate rotation matrix
        if rot_axis == 'x':
            rot_mtx = np.array([[1, 0, 0],
                                [0, cos(angle), -sin(angle)],
                                [0, sin(angle), cos(angle)]])
        elif rot_axis == 'y':
            rot_mtx = np.array([[cos(angle), 0, sin(angle)],
                                [0, 1, 0],
                                [-sin(angle), 0, cos(angle)]])
        elif rot_axis == 'z':
            rot_mtx = np.array([[cos(angle), -sin(angle), 0],
                                [sin(angle), cos(angle), 0],
                                [0, 0, 1]])
        else:
            raise Exception('Unknown axis name, only x, y or z are supported')

        # if size of robot features is greater that rotation matrix shape make rotation matrix part of identity matrix 
        # beginning from the first element, if not just return rotation matrix
        if self.robot_features_size == rot_mtx.shape[0]:
            return rot_axis
        identity_of_size = np.identity(self.robot_features_size)
        identity_of_size[0:rot_mtx.shape[0], 0:rot_mtx.shape[0]] = rot_mtx
        return identity_of_size

    # translation -> move axis by [x,y,z] vector 
    # vect = translation vector
    def translation_matrix(self, vect, axis='', angle=0):
        # rtm -> rotation matrix, 4x4 identity matrix if no angle given to just move matrix in 3D space
        rtm = np.identity(self.robot_features_size) if not axis else self.rotation_matrix(axis, angle)
        # repalce first 3 elems of matrix last column with translated vector x to move matrix by [x,y,z] vector in 3D space
        for x in range(len(vect)):
            rtm[x,len(vect)] = vect[x] 
        return rtm

    # DH_i-1_i = Rt(Z, Oi) * Tr([0, 0, Ei]^T) * Tr([ai, 0, 0]^T) * Rt(X, Li)
    def transformation_matrix(self, theta_i, epsilon_i, a_i, alpha_i):
        rot_mtx_z_theta = self.rotation_matrix('z', theta_i)
        tr_mtx_epsilon = self.translation_matrix([0, 0, epsilon_i])
        tr_mtx_a = self.translation_matrix([a_i, 0, 0])
        rot_mtx_z_alpha = self.rotation_matrix('x', alpha_i)
        dh_i = rot_mtx_z_theta.dot(tr_mtx_epsilon).dot(tr_mtx_a).dot(rot_mtx_z_alpha)
        return np.array(dh_i)

    # combine all matrix operations into forward kinematics
    def forward_kinematics(self, thetas, epsilons, ais, alphas):
        if not all(x == len(thetas) for x in (len(thetas), len(epsilons), len(ais), len(alphas))):
            raise Exception('All homogenous matrix arguments size should be equal each other and robot DOF')

        self.robot_features_size = len(thetas) # set size of operation matrixes
        fw_kine_matrix = [self.transformation_matrix(thetas[0], epsilons[0], ais[0], alphas[0])] # init result with first transformation matrix

        # multiply transformation matrixes one by one
        for elem in range(len(thetas) - 1):
            next_matrix = fw_kine_matrix[elem].dot(self.transformation_matrix(thetas[elem+1], epsilons[elem+1], ais[elem+1], alphas[elem+1]))
            fw_kine_matrix.append(next_matrix)

        return fw_kine_matrix[-1], fw_kine_matrix