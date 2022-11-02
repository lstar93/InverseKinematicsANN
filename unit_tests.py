import unittest
import numpy as np
from math import sin, cos, pi
from robot import ForwardKinematics, Point, InverseKinematics, distance_between_points, get_point_between_points

def resize_rotation_matrix(original_rotation, size):
    if size != original_rotation.shape[0]:
        ident = np.identity(size)
        ident[0:size-1, 0:size-1] = original_rotation
        return ident
    else:
        return original_rotation
        

class forward_kinematics_unittests(unittest.TestCase):

    features_number = 4

    fkine = ForwardKinematics(features_number)

    theta = pi/2
    rot_x_matrix = np.array([[1,0,0],
                            [0,cos(theta),-sin(theta)],
                            [0,sin(theta),cos(theta)]])
    rot_x_matrix = resize_rotation_matrix(rot_x_matrix, 4)

    rot_y_matrix = np.array([[cos(theta),0,sin(theta)],
                            [0,1,0],
                            [-sin(theta),0,cos(theta)]])
    rot_y_matrix = resize_rotation_matrix(rot_y_matrix, 4)

    rot_z_matrix = np.array([[cos(theta),-sin(theta),0],
                            [sin(theta),cos(theta),0],
                            [0,0,1]])
    rot_z_matrix = resize_rotation_matrix(rot_z_matrix, 4)

    translation_matrix = np.array([[1,0,0,1],
                                   [0,1,0,2],
                                   [0,0,1,3],
                                   [0,0,0,1]])

    transformation_matrix_x = fkine.rotation_matrix('x', theta)
    transformation_matrix_x[0,3] = 1
    transformation_matrix_x[1,3] = 2
    transformation_matrix_x[2,3] = 3

    transformation_matrix_y = fkine.rotation_matrix('y', theta)
    transformation_matrix_y[0,3] = 1
    transformation_matrix_y[1,3] = 2
    transformation_matrix_y[2,3] = 3

    transformation_matrix_z = fkine.rotation_matrix('z', theta)
    transformation_matrix_z[0,3] = 1
    transformation_matrix_z[1,3] = 2
    transformation_matrix_z[2,3] = 3

    el_1 = 3
    el_2 = 5
    dh_test_arr = np.matlib.array([[cos(theta),  sin(theta),      0,        el_2 * cos(theta)], 
                                   [sin(theta), -cos(theta),      0,        el_2 * sin(theta)], 
                                   [     0,            0,        -1,        el_1              ], 
                                   [     0,            0,         0,                1         ]])

    # example forward kinematics matrixes
    f_kine_0 = np.matlib.array([[0.5,0.5,0.70710678,3.41421356],
                                [0.5,0.5,-0.70710678,3.41421356],
                                [-0.70710678,0.70710678,0,2],
                                [0,0,0,1]])

    f_kine_1 = np.matlib.array([[-0.5,-0.5,0.70710678,0],
                                [-0.5,-0.5,-0.70710678,0],
                                [0.70710678,-0.70710678,0,6.82842712],
                                [0,0,0,1]])
    f_kine_2 = np.matlib.array([[0.5,0.866025,0,4],
                                [0,0,-1,0],
                                [-0.866025,0.5000,0,2],
                                [0,0,0,1]])

    def test_rotation_matrix(self):
        fkine = ForwardKinematics(self.features_number)
        print(self.rot_x_matrix)
        print(fkine.rotation_matrix('x', pi/2))
        np.testing.assert_array_almost_equal(self.rot_x_matrix, fkine.rotation_matrix('x', pi/2))
        np.testing.assert_array_almost_equal(self.rot_y_matrix, fkine.rotation_matrix('y', pi/2))
        np.testing.assert_array_almost_equal(self.rot_z_matrix, fkine.rotation_matrix('z', pi/2))

    def test_translation_matrix(self):
        fkine = ForwardKinematics(self.features_number)
        np.testing.assert_array_almost_equal(self.translation_matrix, fkine.translation_matrix([1, 2, 3]))
        
    # transformation -> translation + rotation
    def test_transformation_matrix(self):
        fkine = ForwardKinematics(self.features_number)
        np.testing.assert_array_almost_equal(self.transformation_matrix_x, fkine.translation_matrix([1, 2, 3], 'x', pi/2))
        np.testing.assert_array_almost_equal(self.transformation_matrix_y, fkine.translation_matrix([1, 2, 3], 'y', pi/2))
        np.testing.assert_array_almost_equal(self.transformation_matrix_z, fkine.translation_matrix([1, 2, 3], 'z', pi/2))

    def test_prev_to_curr_joint_transform_matrix(self):
        fkine = ForwardKinematics(self.features_number)
        np.testing.assert_array_almost_equal(self.dh_test_arr, fkine.transformation_matrix(pi/2, 3, 5, pi))

    def test_forward_kinematics(self):
        fkine = ForwardKinematics(self.features_number)
        tout, _ = fkine.forward_kinematics([pi/4, pi/4, -pi/4, -pi/4], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.f_kine_0, tout)
        tout1, _ = fkine.forward_kinematics([pi/4, pi/4, pi/4, pi/4], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.f_kine_1, tout1)
        tout2, _ = fkine.forward_kinematics([0, pi/3, -pi/3, -pi/3], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0])
        np.testing.assert_array_almost_equal(self.f_kine_2, tout2)

    def test_point(self):
        p0 = Point([3,3,3])
        p1 = Point([4,6,4])
        pdist = distance_between_points(p0, p1)
        np.testing.assert_almost_equal(pdist, 3.3166247903554) # check first to second distance
        pdist1 = distance_between_points(p1, p0)
        np.testing.assert_almost_equal(pdist1, 3.3166247903554) # check second to first distance
        np.testing.assert_almost_equal(pdist1, pdist) # check both computed distances
        p2 = get_point_between_points(p0, p1, 3) # check slightly less distance point position distance_between_points(p0, p1)
        np.testing.assert_array_almost_equal(np.array([p2.x, p2.y, p2.z]), np.array([3.9045340337332908, 5.713602101199873, 3.9045340337332908]))

    def test_fabrik(self):
        # constants
        dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
        joints_lengths = [2, 2, 2, 2]
        robo_arm_joint_limits = {'x_limits': [0,6], 'y_limits': [-6,6], 'z_limits': [0,6]} # assumed limits
        dest_points = [[2, -2, 4], [1, -4, 5], [1, -2, 3], [1, 3, 1], [2, 2, 4]]
        # kinematics engines
        ikine = InverseKinematics(dh_matrix, joints_lengths, robo_arm_joint_limits)
        fkine = ForwardKinematics(self.features_number)

        for dest_point in dest_points:
            ik_angles = ikine.compute_roboarm_ik('FABRIK', dest_point, 0.001, 100)
            dh_matrix_out = [ik_angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
            fk, _ = fkine.forward_kinematics(*dh_matrix_out)
            forward_dest_point = [fk[0,3], fk[1,3], fk[2,3]]
            max_decimal_error = 3 # set max decimail error to the same accuracy as IK, 3 decmial places
            np.testing.assert_array_almost_equal(np.array(dest_point), np.array(forward_dest_point), max_decimal_error)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(forward_kinematics_unittests('test_rotation_matrix'))
    suite.addTest(forward_kinematics_unittests('test_translation_matrix'))
    suite.addTest(forward_kinematics_unittests('test_transformation_matrix'))
    suite.addTest(forward_kinematics_unittests('test_prev_to_curr_joint_transform_matrix'))
    suite.addTest(forward_kinematics_unittests('test_forward_kinematics'))
    suite.addTest(forward_kinematics_unittests('test_point'))
    suite.addTest(forward_kinematics_unittests('test_fabrik'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())