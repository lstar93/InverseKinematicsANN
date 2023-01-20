""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

# pylint: disable=W0105 # suppress unnecesary strings in code
# pylint: disable=W0511 # suppress TODOs
# pylint: disable=W1401 # string constant might be missing r prefix
# pylint: disable=W0105 # string has no effect
# pylint: disable=C0413 # imports should be placed at the top of the module

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

TODO:
1. read robot configuration (with angle limits) from file
'''

import argparse
from math import pi
import pandas as pd
from inverse_kinematics import AnnInverseKinematics, FabrikInverseKinematics
from plot import plot_points_3d, plot_joint_points_3d
from position_generator import TrainingDataGenerator
from forward_kinematics import ForwardKinematics

# 6 DOF robot DH matrix, links lengths, workspace and joints limits
dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]}
links_lengths = [2, 2, 2, 2]


def cli_ikine(parser):
    """ Inverse kinematics CLI """
    parser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                            help='select inverse kinematics engine, Neural Network or Fabrik')

    known_args, _ = parser.parse_known_args()
    if known_args.method == 'ann':
        parser.add_argument('--model', type=str, required=True,
                    help='select .h5 file with saved mode, required only if ann was choosed')

    parser.add_argument('--points', type=str, required=True,
                            help='.csv file name with stored trajectory points')

    parser.add_argument('--to-file', type=str)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    ikine_method = args.method
    points_file = args.points
    points = pd.read_csv(points_file).values.tolist()

    joint_angles = []
    if ikine_method == 'ann':
        model = args.model
        ik_engine = AnnInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)
        ik_engine.load_model(model)
        joint_angles = [ik_engine.ikine([pos]) for pos in points]

    elif ikine_method == 'fabrik':
        ik_engine = FabrikInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)
        joint_angles = [ik_engine.ikine(pos) for pos in points]

    if args.to_file is not None:
        pd.DataFrame(joint_angles,
                     columns=['theta1', 'theta2', 'theta3', 'theta4'])\
                     .to_csv(args.to_file, index=False)

    if args.verbose:
        print(joint_angles)

    return joint_angles, points


def cli_gen_data(parser):
    """ Simple CLI to generate .csv with trajectories """
    # parser = argparse.ArgumentParser(prog='positions_generator')
    parser.add_argument('--shape', required=True, type=str,
                            choices=['circle', 'cube', 'cube_random',\
                                        'random', 'spring', 'random_dist'],
                            help='select which shape should be generated')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--to-file', type=str)

    known_args, _ = parser.parse_known_args()
    verbose = known_args.verbose
    filename = known_args.to_file
    points = []

    # --generate-data --shape circle --radius 3 --samples 20 --centre '1,11,2' --verbose
    if known_args.shape == 'circle':
        parser.add_argument('--radius', required=True, type=int)
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--centre', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        radius = known_args.radius
        samples = known_args.samples
        centre = [int(pos) for pos in (known_args.centre.split(','))]
        points = TrainingDataGenerator.circle(radius, samples, centre)
        if verbose:
            print(radius, samples, centre)
            plot_points_3d(points)

    # --generate-data --shape cube --step 0.75 --dim '2,3,4' --start '1,2,3' --verbose
    # --generate-data --shape cube_random --step 0.75 --dim '2,3,4' --start '1,2,3' --verbose
    def cube(generator):
        parser.add_argument('--step', required=True, type=float)
        parser.add_argument('--dim', required=True, type=str)
        parser.add_argument('--start', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        step = known_args.step
        dim = [int(pos) for pos in (known_args.dim.split(','))]
        start = [int(pos) for pos in (known_args.start.split(','))]
        points = generator(step, *dim, start)
        if verbose:
            print(step, dim, start)
            plot_points_3d(points)

    if known_args.shape == 'cube':
        cube(TrainingDataGenerator.cube)
    elif known_args.shape == 'cube_random':
        cube(TrainingDataGenerator.cube_random)

    # --generate-data --shape random --limits '0,3;0,4;0,5' --samples 20 --verbose
    if known_args.shape == 'random':
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--limits', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        samples = known_args.samples
        limits = list(known_args.limits.split(';'))
        limits_dict = {'x': [int(pos) for pos in (limits[0].split(','))],
                    'y': [int(pos) for pos in (limits[1].split(','))],
                    'z': [int(pos) for pos in (limits[2].split(','))]}
        points = TrainingDataGenerator.random(samples, limits_dict)
        if verbose:
            print(samples, limits_dict)
            plot_points_3d(points)

    # --generate-data --shape spring --samples 50 --dim '2,3,6' --verbose
    if known_args.shape == 'spring':
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--dim', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        samples = known_args.samples
        dim = [int(pos) for pos in (known_args.dim.split(','))]
        points = TrainingDataGenerator.spring(samples, *dim)
        if verbose:
            print(samples, dim)
            plot_points_3d(points)

    # --generate-data --shape random_dist --dist 'normal' \
    #   --samples 100 --std_dev 0.35 --limits '0,3;0,4;0,5' --verbose
    # --generate-data --shape random_dist --dist 'uniform' \
    #   --samples 100 --std_dev 0.35 --limits '0,3;0,4;0,5' --verbose
    # --generate-data --shape random_dist --dist 'random' \
    #   --samples 100 --std_dev 0.35 --limits '0,3;0,4;0,5' --verbose
    if known_args.shape == 'random_dist':
        parser.add_argument('--dist', required=True, type=str,
                            choices=['normal', 'uniform', 'random'])
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--std_dev', required=True, type=float)
        parser.add_argument('--limits', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        dist = known_args.dist
        samples = known_args.samples
        std_dev = known_args.std_dev
        limits = list(known_args.limits.split(';'))
        limits_dict = {'x': [int(pos) for pos in (limits[0].split(','))],
                    'y': [int(pos) for pos in (limits[1].split(','))],
                    'z': [int(pos) for pos in (limits[2].split(','))]}
        points = TrainingDataGenerator.random_distribution(samples, limits_dict, dist, std_dev)
        if verbose:
            print(samples, std_dev, limits_dict)
            plot_points_3d(points)

    if filename is not None:
        pd.DataFrame(points, columns=['x', 'y', 'z']).to_csv(filename, index=False)


if __name__ == '__main__':
    cliparser = argparse.ArgumentParser(prog='cli')

    '''
    parser.add_argument('--inverse-kine', required = True, action='store_true',
                            help='use ANN or Fabrik to compute robot inverse kinematics')
    parser.add_argument('--generate-data', action='store_true',
                            help='use one of the available data generators to create tracjectory')

    cliparser.add_argument('--execute', required = True, choices=['inverse_kine', 'generate_data'],
                            help='use ANN or Fabrik to compute robot inverse kinematics \
                                or choose generator and generate trajectory to csv file')
    '''
    group = cliparser.add_mutually_exclusive_group()
    group.add_argument('--inverse-kine', action='store_true')
    group.add_argument('--generate-data', action='store_true')

    cli_known_args, _ = cliparser.parse_known_args()

    if cli_known_args.inverse_kine is False and cli_known_args.generate_data is False:
        cliparser.error('Operation --inverse-kine or --generate-data must be set')

    if cli_known_args.inverse_kine:
        angles_ik, input_points = cli_ikine(cliparser)

        # use ForwardKinematics to check predictions
        predicted_points = []
        fkine = ForwardKinematics()
        for angles in angles_ik:
            dh_matrix_out = [angles, *dh_matrix[1:]]
            fk, _ = fkine.fkine(*dh_matrix_out)
            predicted_points.append([fk[0,3], fk[1,3], fk[2,3]])

        plot_joint_points_3d(predicted_points, input_points)

    elif cli_known_args.generate_data:
        cli_gen_data(cliparser)
