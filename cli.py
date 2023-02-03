""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

# pylint: disable=W0105 # suppress unnecesary strings in code
# pylint: disable=W0511 # suppress TODOs
# pylint: disable=W1401 # string constant might be missing r prefix
# pylint: disable=W0105 # string has no effect
# pylint: disable=C0413 # imports should be placed at the top of the module

import os
import sys
import argparse
from abc import ABC, abstractmethod
import pandas as pd
from kinematics.forward import ForwardKinematics
from kinematics.inverse import AnnInverseKinematics, FabrikInverseKinematics
from plot.plot import Plotter
from robot.position_generator import TrainingDataGenerator
from robot.robot import SixDOFRobot as Robot
from robot.robot import OutOfRobotReachException


class Command(ABC):
    """ CLI command base """
    @abstractmethod
    def generate(self, parser):
        """ Generate data """

    @abstractmethod
    def example(self):
        """ Get example command """


class CircleCommand(Command):
    """ Circle points CLI generator """
    COMMAND = 'circle'

    def __init__(self):
        self.generator = TrainingDataGenerator.circle

    def generate(self, parser):
        """ Generate data """
        parser.add_argument('--radius', required=True, type=float)
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--centre', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        radius = known_args.radius
        samples = known_args.samples
        centre = [float(pos) for pos in (known_args.centre.split(','))]
        return (radius, samples, centre), self.generator(radius, samples, centre)

    def example(self):
        return "--generate-data --shape circle --radius 3 --samples 20 --centre '1,11,2'"


class CubeCommand(Command):
    """ Cube points CLI generator """
    COMMAND = 'cube'

    def __init__(self):
        self.generator = TrainingDataGenerator.cube

    def generate(self, parser):
        parser.add_argument('--step', required=True, type=float)
        parser.add_argument('--dim', required=True, type=str)
        parser.add_argument('--start', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        step = known_args.step
        dim = [float(pos) for pos in (known_args.dim.split(','))]
        start = [float(pos) for pos in (known_args.start.split(','))]
        return (step, dim, start), self.generator(step, *dim, start)

    def example(self):
        return "--generate-data --shape cube --step 0.75 --dim '2,3,4' --start '1,2,3'"


class CubeRandomCommand(CubeCommand):
    """ Random cube points CLI generator """
    COMMAND = 'cube_random'

    def __init__(self):
        super().__init__()
        self.generator = TrainingDataGenerator.cube_random

    def example(self):
        return "--generate-data --shape cube_random --step 0.75 --dim '2,3,4' --start '1,2,3'"


class RandomCommand(Command):
    """ Random points CLI generator """
    COMMAND = 'random'

    def __init__(self):
        self.generator = TrainingDataGenerator.random

    def generate(self, parser):
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--limits', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        samples = known_args.samples
        limits = list(known_args.limits.split(';'))
        limits_dict = {'x': [float(pos) for pos in (limits[0].split(','))],
                       'y': [float(pos) for pos in (limits[1].split(','))],
                       'z': [float(pos) for pos in (limits[2].split(','))]}
        return (samples, limits_dict), self.generator(samples, limits_dict)

    def example(self):
        return "--generate-data --shape random --limits '0,3;0,4;0,5' --samples 20"


class SpringCommand(Command):
    """ Spring points CLI generator """
    COMMAND = 'spring'

    def __init__(self):
        self.generator = TrainingDataGenerator.spring

    def generate(self, parser):
        parser.add_argument('--samples', required=True, type=int)
        parser.add_argument('--dim', required=True, type=str)
        known_args, _ = parser.parse_known_args()
        samples = known_args.samples
        dim = [float(pos) for pos in (known_args.dim.split(','))]
        return (samples, dim), self.generator(samples, *dim)

    def example(self):
        return "--generate-data --shape spring --samples 50 --dim '2,3,6'"


class RandomDistributionCommand(Command):
    """ Random distribution points CLI generator """
    COMMAND = 'random_dist'

    def __init__(self):
        self.generator = TrainingDataGenerator.random_distribution

    def generate(self, parser):
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
        limits_dict = {'x': [float(pos) for pos in (limits[0].split(','))],
                       'y': [float(pos) for pos in (limits[1].split(','))],
                       'z': [float(pos) for pos in (limits[2].split(','))]}
        return (samples, std_dev, limits_dict), self.generator(samples, limits_dict, dist, std_dev)

    def example(self):
        return "--generate-data --shape random_dist --dist 'normal'\
        ' --samples 100 --std_dev 0.35 --limits '0,3;0,4;0,5'"


class CLIData:
    """ Simple CLI to generate .csv with trajectories """

    def __init__(self, parser):
        """ Init parser and possible commands """
        self.commands = {
            CircleCommand.COMMAND: CircleCommand(),
            CubeCommand.COMMAND: CubeCommand(),
            CubeRandomCommand.COMMAND: CubeRandomCommand(),
            RandomCommand.COMMAND: RandomCommand(),
            SpringCommand.COMMAND: SpringCommand(),
            RandomDistributionCommand.COMMAND: RandomDistributionCommand()
        }
        self.parser = parser

    def cli(self):
        """ Parse CLI """
        self.parser.add_argument('--shape', required=True, type=str,
                                choices=list(self.commands.keys()),
                                help='select which shape should be generated')

        self.parser.add_argument('--verbose', action='store_true')
        self.parser.add_argument('--example', action='store_true')
        self.parser.add_argument('--to-file', type=str)

        known_args, _ = self.parser.parse_known_args()
        verbose = known_args.verbose
        filename = known_args.to_file

        if known_args.example:
            print(self.commands[known_args.shape].example())
        else:
            data, points = self.commands[known_args.shape].generate(self.parser)

            if filename is not None:
                pd.DataFrame(points, columns=['x', 'y', 'z']).to_csv(filename, index=False)

            if verbose:
                print(data)
                Plotter.plot_points_3d(points)


class CLIIkine:
    """ Inverse kinematics CLI """

    def __init__(self, parser):
        """ Init parser """
        self.parser = parser

    def __forward_kinematics(self, angles):
        """ Calcualte forward kinematics """
        fkine_points = []
        fkine = ForwardKinematics(Robot.dh_matrix)
        for pos_angles in angles:
            fkp, _ = fkine.fkine(pos_angles)
            fkine_points.append([fkp[0,3], fkp[1,3], fkp[2,3]])
        return fkine_points

    def cli(self):
        """ Inverse kinematics CLI """
        self.parser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                                help='select inverse kinematics method, Neural Network or Fabrik')
        self.parser.add_argument('--list-models', action='store_true')

        known_args, _ = self.parser.parse_known_args()
        if known_args.list_models:
            ann_models = os.listdir('models')
            print('Available models:')
            for ann_model in ann_models:
                print(ann_model)
                sys.exit(0)

        if known_args.method == 'ann':
            self.parser.add_argument('--model', type=str, required=True,
            help='select .h5 file with saved model, required only if ann ikine method was choosed')

        self.parser.add_argument('--points', type=str, required=True,
                                help='.csv file name with stored trajectory points')

        self.parser.add_argument('--to-file', type=str)
        self.parser.add_argument('--verbose', action='store_true')
        self.parser.add_argument('--show-path', action='store_true')
        self.parser.add_argument('--separate-plots', action='store_true')

        args = self.parser.parse_args()

        points = pd.read_csv(args.points).values.tolist()

        joint_angles = []
        if args.method == 'ann':
            model = args.model
            ik_engine = AnnInverseKinematics(Robot.dh_matrix,
                                            Robot.links_lengths,
                                            Robot.effector_workspace_limits)
            ik_engine.load_model(model)
            joint_angles = ik_engine.ikine(points)

        elif args.method == 'fabrik':
            ik_engine = FabrikInverseKinematics(Robot.dh_matrix,
                                                Robot.links_lengths,
                                                Robot.effector_workspace_limits)
            joint_angles = ik_engine.ikine(points)

        if args.to_file is not None:
            pd.DataFrame(joint_angles,
                        columns=['theta1', 'theta2', 'theta3', 'theta4'])\
                        .to_csv(args.to_file, index=False)

        if args.verbose:
            predicted_points = self.__forward_kinematics(joint_angles)
            if not args.separate_plots:
                Plotter.plot_joint_points_3d(predicted_points, points, args.show_path)
            else:
                Plotter.plot_points_3d(points, args.show_path, dot_color='b')
                Plotter.plot_points_3d(predicted_points, args.show_path, dot_color='r')
            print(joint_angles)


if __name__ == '__main__':
    # Choose CLI handler
    cliparser = argparse.ArgumentParser(prog='cli')
    group = cliparser.add_mutually_exclusive_group()
    group.add_argument('--inverse-kine', action='store_true')
    group.add_argument('--generate-data', action='store_true')

    cli_known_args, _ = cliparser.parse_known_args()

    if cli_known_args.inverse_kine is False and cli_known_args.generate_data is False:
        cliparser.error('Operation --inverse-kine or --generate-data must be choosed')

    # Handle inverse kinematics CLI
    if cli_known_args.inverse_kine:
        try:
            cli_ikine = CLIIkine(cliparser)
            cli_ikine.cli()
        except OutOfRobotReachException as kine_exception:
            print(str(kine_exception))
            sys.exit(0)

    # or handle inverse kinematics data generators
    elif cli_known_args.generate_data:
        cli_data_generators = CLIData(cliparser)
        cli_data_generators.cli()
