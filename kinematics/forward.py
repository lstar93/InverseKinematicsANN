""" Planar robot forward kinematics  """
#!/usr/bin/env python

from math import sin, cos, pi
import numpy as np

# supress printing enormous small numbers like 0.123e-16
np.set_printoptions(suppress=True)


class ForwardKinematics:
    """ Planar robotic arm forward kinematics """
    def __init__(self, dh_matrix):
        if not all(len(x) == len(dh_matrix[0]) for x in dh_matrix):
            raise Exception('All homogenous matrix elements should have equal size')
        self.dh_matrix = dh_matrix
        self.thetas, self.epsilons, self.ais, self.alphas = self.dh_matrix
        # number of features is equal to lenghts of one of dh parameters lists
        self.no_of_features = len(self.thetas)

    def __rotation_matrix(self, rot_axis, angle):
        """ Create rotation matrix around axis x, y or z """
        if (angle < -2*pi) or (angle > 2*pi):
            raise Exception('Error, angle limits are (-2pi, 2pi)')
        if self.no_of_features < 3:
            raise Exception('Error, rotation matrix size must be 3 or greater')

        # Generate rotation matrix
        if rot_axis == 'x':
            rot_mtx = np.array([[1,           0,          0],
                                [0, cos(angle), -sin(angle)],
                                [0, sin(angle), cos(angle)]])
        elif rot_axis == 'y':
            rot_mtx = np.array([[cos(angle),  0, sin(angle)],
                                [0,           1,          0],
                                [-sin(angle), 0, cos(angle)]])
        elif rot_axis == 'z':
            rot_mtx = np.array([[cos(angle), -sin(angle), 0],
                                [sin(angle), cos(angle),  0],
                                [0,           0,          1]])
        else:
            raise Exception('Unknown axis name, only x, y or z are supported')

        # if size of robot features is greater than rotation matrix shape size
        # make rotation matrix part of identity matrix beginning from the first element
        if self.no_of_features == rot_mtx.shape[0]:
            return rot_axis
        identity_of_size = np.identity(self.no_of_features)
        identity_of_size[0:rot_mtx.shape[0], 0:rot_mtx.shape[0]] = rot_mtx
        return identity_of_size

    def __translation_matrix(self, vect, axis=None, angle=0):
        """ Create translation matrix -> move it by vector """
        # rtm -> rotation matrix, 4x4 identity matrix
        # if no angle given just move matrix
        if not axis:
            rtm = np.identity(self.no_of_features)
        else:
            self.__rotation_matrix(axis, angle)
        # repalce first 3 elems of last column with transposed vector x
        # to move matrix by [x,y,z] vector
        for index, _ in enumerate(vect):
            rtm[index, len(vect)] = vect[index]
        return rtm

    # DH_i-1_i = Rt(Z, Oi) * Tr([0, 0, Ei]^T) * Tr([ai, 0, 0]^T) * Rt(X, Li)
    def __transformation_matrix(self, theta_i, epsilon_i, a_i, alpha_i):
        """ Create forward kinematics transformation matrix """
        rot_mtx_z_theta = self.__rotation_matrix('z', theta_i)
        tr_mtx_epsilon = self.__translation_matrix([0, 0, epsilon_i])
        tr_mtx_a = self.__translation_matrix([a_i, 0, 0])
        rot_mtx_z_alpha = self.__rotation_matrix('x', alpha_i)
        dh_i = rot_mtx_z_theta.dot(tr_mtx_epsilon).dot(tr_mtx_a).dot(rot_mtx_z_alpha)
        return np.array(dh_i)

    # combine all matrix operations into forward kinematics
    def fkine(self, angles):
        """ Calculate robotic arm forward kinematics """

        # replace initial thetas with desired joints angles
        self.thetas = angles

        # init result with first transformation matrix
        fw_kine_matrix = [self.__transformation_matrix(
                            self.thetas[0], self.epsilons[0],
                            self.ais[0], self.alphas[0])]

        # multiply transformation matrixes one by one
        for elem in range(self.no_of_features - 1):
            next_matrix = fw_kine_matrix[elem].dot(
                self.__transformation_matrix(
                    self.thetas[elem+1], self.epsilons[elem+1],
                    self.ais[elem+1], self.alphas[elem+1])
                )
            # create chain of multiplied matrixes
            fw_kine_matrix.append(next_matrix)

        return fw_kine_matrix[-1], fw_kine_matrix
