""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

# pylint: disable=W0105 # suppress unnecesary strings in code

import argparse
from math import pi
import pandas as pd
from inverse_kinematics import AnnInverseKinematics, FabrikInverseKinematics
from plot import plot_points_3d
from position_generator import TrainingDataGenerator

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

    parser.add_argument('--plot', action='store_true',
                            help='add to plot results')

    args = parser.parse_args()

    # print(args.inverse_kine)
    ikine_method = args.method
    model = args.model
    points_file = args.points

    # 6 DOF robot DH matrix, links lengths, workspace and joints limits
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
    effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]}
    links_lengths = [2, 2, 2, 2]

    '''
    if ikine_method == 'ann':
        ik_engine = AnnInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)
        ik_engine.load_model(model)
    elif ikine_method == 'fabrik':
        ik_engine = FabrikInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

    points_df = pd.read_csv(points_file)
    # convert dataframe to list of points
    angles = [ik_engine.ikine(pos) for pos in points_df]
    '''

def cli_gen_data(parser):
    """ Simple CLI to generate .csv with trajectories """
    # parser = argparse.ArgumentParser(prog='positions_generator')
    parser.add_argument('--shape', required=True, type=str,
                            choices=['circle', 'cube', 'cube_random',\
                                        'random', 'spring', 'random_dist'],
                            help='select which shape should be generated')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--filename', type=str)

    known_args, _ = parser.parse_known_args()
    verbose = known_args.verbose
    filename = known_args.filename
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

    return points


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
        cliparser.error('Operation either --inverse-kine or --generate-data must be set')

    if cli_known_args.inverse_kine:
        cli_ikine(cliparser)
    elif cli_known_args.generate_data:
        cli_gen_data(cliparser)
