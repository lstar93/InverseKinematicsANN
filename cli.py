""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

# pylint: disable=W0105 # suppress unnecesary strings in code
# pylint: disable=W0511 # suppress TODOs
# pylint: disable=W1401 # string constant might be missing r prefix
# pylint: disable=W0105 # string has no effect
# pylint: disable=C0413 # imports should be placed at the top of the module


import sys
import argparse
from abc import ABC, abstractmethod
import pandas as pd
from kinematics.forward import ForwardKinematics
from kinematics.inverse import AnnInverseKinematics, FabrikInverseKinematics
from plot.plot import Plotter
from robot.position_generator import TrainingDataGenerator
from robot.robot import OutOfRobotReachException
from robot.robot import SixDOFRobot as Robot


class Command(ABC):
    """ CLI command base """
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    @abstractmethod
    def execute(self):
        """ Generate data """

    @abstractmethod
    def example(self):
        """ Get example command """


class ShapeCommand(Command):
    """ Shape generator subclass """
    def verbose(self, data, points):
        """ Print output and plot shape """
        print(data)
        Plotter.plot_points_3d(points)

    def save_to_csv(self, data, filename):
        """ Save data to csv file """
        pd.DataFrame(data,
                     columns=['x', 'y', 'z'])\
                     .to_csv(filename, index=False)


class IkineCommand(Command):
    """ Inverse kinematics subclass """
    def __forward_kinematics(self, angles):
        """ Calcualte forward kinematics """
        fkine_points = []
        fkine = ForwardKinematics(Robot.dh_matrix)
        for pos_angles in angles:
            fkp, _ = fkine.fkine(pos_angles)
            fkine_points.append([fkp[0,3], fkp[1,3], fkp[2,3]])
        return fkine_points

    def verbose(self, points, joint_angles, separate_plots, show_path):
        """ Print output and plot prediction """
        predicted_points = self.__forward_kinematics(joint_angles)
        if separate_plots:
            Plotter.plot_points_3d(points, show_path, dot_color='b')
            Plotter.plot_points_3d(predicted_points, show_path, dot_color='r')
        else:
            Plotter.plot_joint_points_3d(predicted_points, points, show_path)

        print(joint_angles)

    def save_to_csv(self, data, filename):
        """ Save data to csv file """
        pd.DataFrame(data,
                     columns=['theta1', 'theta2', 'theta3', 'theta4'])\
                     .to_csv(filename, index=False)

# Data generators commands

class CircleCommand(ShapeCommand):
    """ Circle generator """
    COMMAND = 'circle'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.circle

    def execute(self):
        """ Generate data """
        self.parser.add_argument('--radius', required=True, type=float)
        self.parser.add_argument('--samples', required=True, type=int)
        self.parser.add_argument('--center', required=True, type=str)
        known_args, _ = self.parser.parse_known_args()
        radius = known_args.radius
        samples = known_args.samples
        center = [float(pos) for pos in (known_args.center.split(','))]
        points = self.generator(radius, samples, center)
        if known_args.verbose:
            self.verbose((radius, samples, center), points)
        if known_args.to_file is not None:
            self.save_to_csv(points, known_args.to_file)

    def example(self):
        return "--generate-data --shape circle --radius 3 --samples 20 --center 1,5,2"


class CubeCommand(ShapeCommand):
    """ Cube generator """
    COMMAND = 'cube'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.cube

    def execute(self):
        self.parser.add_argument('--step', required=True, type=float)
        self.parser.add_argument('--dim', required=True, type=str)
        self.parser.add_argument('--start', required=True, type=str)
        known_args, _ = self.parser.parse_known_args()
        step = known_args.step
        dim = [float(pos) for pos in (known_args.dim.split(','))]
        start = [float(pos) for pos in (known_args.start.split(','))]
        points = self.generator(step, *dim, start)
        if known_args.verbose:
            self.verbose((step, dim, start), points)
        if known_args.to_file is not None:
            self.save_to_csv(points, known_args.to_file)

    def example(self):
        return "--generate-data --shape cube --step 0.75 --dim 2,3,4 --start 1,2,3"


class CubeRandomCommand(CubeCommand):
    """ Random cube generator """
    COMMAND = 'cube_random'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.cube_random

    def example(self):
        return "--generate-data --shape cube_random --step 0.75 --dim 2,3,4 --start 1,2,3"


class RandomCommand(ShapeCommand):
    """ Random generator """
    COMMAND = 'random'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.random

    def execute(self):
        self.parser.add_argument('--samples', required=True, type=int)
        self.parser.add_argument('--limits', required=True, type=str)
        known_args, _ = self.parser.parse_known_args()
        samples = known_args.samples
        limits = list(known_args.limits.split(';'))
        limits_dict = {'x': [float(pos) for pos in (limits[0].split(','))],
                       'y': [float(pos) for pos in (limits[1].split(','))],
                       'z': [float(pos) for pos in (limits[2].split(','))]}
        points = self.generator(samples, limits_dict)
        if known_args.verbose:
            self.verbose((samples, limits_dict), points)
        if known_args.to_file is not None:
            self.save_to_csv(points, known_args.to_file)

    def example(self):
        return "--generate-data --shape random --limits 0,3;0,4;0,5 --samples 20"


class SpringCommand(ShapeCommand):
    """ Spring generator """
    COMMAND = 'spring'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.spring

    def execute(self):
        self.parser.add_argument('--samples', required=True, type=int)
        self.parser.add_argument('--dim', required=True, type=str)
        known_args, _ = self.parser.parse_known_args()
        samples = known_args.samples
        dim = [float(pos) for pos in (known_args.dim.split(','))]
        points = self.generator(samples, *dim)
        if known_args.verbose:
            self.verbose((samples, *dim), points)
        if known_args.to_file is not None:
            self.save_to_csv(points, known_args.to_file)

    def example(self):
        return "--generate-data --shape spring --samples 50 --dim 2,3,6"


class RandomDistributionCommand(ShapeCommand):
    """ Random distribution generator """
    COMMAND = 'random_dist'

    def __init__(self, parser):
        super().__init__(parser)
        self.generator = TrainingDataGenerator.random_distribution

    def execute(self):
        self.parser.add_argument('--dist', required=True, type=str,
                            choices=['normal', 'uniform', 'random'])
        self.parser.add_argument('--samples', required=True, type=int)
        self.parser.add_argument('--std_dev', required=True, type=float)
        self.parser.add_argument('--limits', required=True, type=str)
        known_args, _ = self.parser.parse_known_args()
        dist = known_args.dist
        samples = known_args.samples
        std_dev = known_args.std_dev
        limits = list(known_args.limits.split(';'))
        limits_dict = {'x': [float(pos) for pos in (limits[0].split(','))],
                       'y': [float(pos) for pos in (limits[1].split(','))],
                       'z': [float(pos) for pos in (limits[2].split(','))]}
        points = self.generator(samples, limits_dict, dist, std_dev)
        if known_args.verbose:
            self.verbose((samples, limits_dict, dist, std_dev), points)
        if known_args.to_file is not None:
            self.save_to_csv(points, known_args.to_file)

    def example(self):
        return "--generate-data --shape random_dist --dist normal "\
        "--samples 100 --std_dev 0.35 --limits 0,3;0,4;0,5"

# Kinematics commands

class InverseKineAnnCommand(IkineCommand):
    """ Calculate inverse kinematics using ANN """
    COMMAND = 'ann'

    def execute(self):
        """ Execute command """
        self.parser.add_argument('--model', type=str, required=True,
        help='select saved mode .h5 filename')
        known_args, _ = self.parser.parse_known_args()
        model = known_args.model
        points = pd.read_csv(known_args.points).values.tolist()
        ik_engine = AnnInverseKinematics(Robot.dh_matrix,
                                         Robot.links_lengths,
                                         Robot.effector_workspace_limits)
        ik_engine.load_model(model)

        try:
            joint_angles = ik_engine.ikine(points)
        except (OutOfRobotReachException, ValueError) as kine_exception:
            print(str(kine_exception))
            return

        if known_args.verbose:
            self.verbose(points, joint_angles,
                known_args.separate_plots, known_args.show_path)

        if known_args.to_file is not None:
            self.save_to_csv(joint_angles, known_args.to_file)

    def example(self):
        return "--inverse-kine --method ann --model model_filename.h5 --points filename.csv"


class InverseKineFabrikCommand(IkineCommand):
    """ Calculate inverse kinematics using Fabrik """
    COMMAND = 'fabrik'

    def execute(self):
        """ Execute command """
        known_args, _ = self.parser.parse_known_args()
        points = pd.read_csv(known_args.points).values.tolist()
        ik_engine = FabrikInverseKinematics(Robot.dh_matrix,
                                            Robot.links_lengths,
                                            Robot.effector_workspace_limits)
        try:
            joint_angles = ik_engine.ikine(points)
        except (OutOfRobotReachException, ValueError) as kine_exception:
            print(str(kine_exception))
            return

        if known_args.verbose:
            self.verbose(points, joint_angles,
                known_args.separate_plots, known_args.show_path)

        if known_args.to_file is not None:
            self.save_to_csv(joint_angles, known_args.to_file)

    def example(self):
        return "--inverse-kine --method fabrik --points filename.csv"

# Executor

class CommandExecutor():
    """ Command executor """

    def execute(self, command: Command):
        """ Execute command """
        command.execute()
        sys.exit(0)

    def example(self, command: Command):
        """ List command example and exit """
        print(command.example())
        sys.exit(0)


class CLI:
    """ CLI class """
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='cli')
        self.executor = CommandExecutor()
        self.data_commands = {
            CircleCommand.COMMAND: CircleCommand(self.parser),
            CubeCommand.COMMAND: CubeCommand(self.parser),
            CubeRandomCommand.COMMAND: CubeRandomCommand(self.parser),
            RandomCommand.COMMAND: RandomCommand(self.parser),
            SpringCommand.COMMAND: SpringCommand(self.parser),
            RandomDistributionCommand.COMMAND: RandomDistributionCommand(self.parser)
        }
        self.ikine_commands = {
            InverseKineAnnCommand.COMMAND: InverseKineAnnCommand(self.parser),
            InverseKineFabrikCommand.COMMAND: InverseKineFabrikCommand(self.parser)
        }

    def __data_cli(self):
        """ Data generator CLI """
        self.parser.add_argument('--shape', required=True, type=str,
                                choices=list(self.data_commands.keys()),
                                help='select which shape should be generated')
        self.parser.add_argument('--example', action='store_true')

        known_args, _ = self.parser.parse_known_args()

        if known_args.example:
            self.executor.example(self.data_commands[known_args.shape])

        self.parser.add_argument('--verbose', action='store_true')
        self.parser.add_argument('--to-file', type=str)

        self.executor.execute(self.data_commands[known_args.shape])

    def __ikine_cli(self):
        """ Inverse kinematics CLI """
        self.parser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                                help='select inverse kinematics method, Neural Network or Fabrik')
        self.parser.add_argument('--example', action='store_true')

        known_args, _ = self.parser.parse_known_args()

        if known_args.example:
            self.executor.example(self.ikine_commands[known_args.method])

        self.parser.add_argument('--points', type=str, required=True,
                                help='.csv file name with stored trajectory points')
        self.parser.add_argument('--to-file', type=str)
        self.parser.add_argument('--verbose', action='store_true')
        self.parser.add_argument('--show-path', action='store_true')
        self.parser.add_argument('--separate-plots', action='store_true')

        self.executor.execute(self.ikine_commands[known_args.method])

    def cli(self, args = None):
        """ cli method """
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--inverse-kine', action='store_true')
        group.add_argument('--generate-data', action='store_true')

        cliargs = sys.argv[1:] if args is None else args
        cli_known_args, _ = self.parser.parse_known_args(cliargs)

        if cli_known_args.inverse_kine is False and cli_known_args.generate_data is False:
            self.parser.error('Operation --inverse-kine or --generate-data must be choosed')

        # Handle inverse kinematics CLI
        if cli_known_args.inverse_kine:
            self.__ikine_cli()
        # or handle inverse kinematics data generators
        elif cli_known_args.generate_data:
            self.__data_cli()

def main():
    """ main function """
    cli = CLI()
    cli.cli()

if __name__ == '__main__':
    main()
