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

from position_generator import *
from forward_kinematics import *
from inverse_kinematics import *
from plot import *

# Use existing inverse kinematics engine (FABRIK) to compute joints angles to test model
def compute_angles(data, ikine_engine):
    angles = []
    for dest_point in data:
        try:
            angles.append(ikine_engine.compute_roboarm_ik('FABRIK', dest_point, 0.001, 100))
        except Exception as e:
            print('Excpetion in compute_angles: {}'.format(e))
    return angles

# test ANN
if __name__ == '__main__':

    # 6 DOF robot DH matrix
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]

    # assumed links lengths
    links_lengths = [2, 2, 2, 2]

    # workspace and joint angles limits
    joints_angles_limits = {'theta_1': [-pi/2,pi/2], 'theta_2': [-pi/4,pi/2], 'theta_3': [-pi/2,pi/2], 'theta_4': [-pi/2,pi/2]} # assumed joints angles limits
    effector_workspace_limits = {'x_limits': [1,6], 'y_limits': [-6,6], 'z_limits': [0,6]} # assumed limits

    # inverse kinematics engine
    ikine = InverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

    # forward kinematics
    fkine = ForwardKinematics()
    ik_angles = [] # computed joints angles

    # random positions generator class -> used to train model
    generator = RoboarmPositionsGenerator()

    try:
        ann = ANN(joints_angles_limits, effector_workspace_limits, dh_matrix)

        limits = {'x_limits': [1,4], 'y_limits': [-4,4], 'z_limits': [0,4]} # assumed limits
        # positions_samples = generator.random(no_of_samples = 300, limits = limits, distribution='random')
        # positions_samples = generator.cube(step_size = 5, limits = limits)
        # angles_features = compute_angles(positions_samples, ikine)

        # model test trajectory datasets
        circle_samples_test = generator.circle(radius = 1, no_of_samples = 20, position = [1,3,1])
        cube_samples_test = generator.cube(step_size = 2, limits = limits)

        # print/plot learn points dataset
        # plot_points_cloud([Point([*elem]) for elem in positions_samples])

        # epochs=2000
        # ann.train_model(epochs, positions_samples, angles_features) # random data

        # from keras.models import load_model
        ann.load_model('roboarm_model_1664488076-610064.h5')

        # print/plot test points dataset
        test_points = [Point([*elem]) for elem in cube_samples_test]
        plot_points_cloud(test_points)
        print('sample: ' + str(np.array(cube_samples_test)))

        # ANN
        print("ANN")
        predicted_points = []
        ik_angles_ann = ann.predict_ik(cube_samples_test).tolist()

        # compute FK to check ANN IK
        for angles in ik_angles_ann:
            dh_matrix_out = [angles, [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
            fk, _ = fkine.forward_kinematics(*dh_matrix_out)
            predicted_points.append(Point([fk[0,3], fk[1,3], fk[2,3]]))

        # print/plot predicted points
        plot_points_cloud(predicted_points)
        print('predicted: ' + str(np.array(predicted_points)))

        print(np.array(cube_samples_test) - np.array(predicted_points))

        # ann.save_model()

    except Exception as e:
        print(str(e))