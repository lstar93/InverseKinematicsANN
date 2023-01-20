""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

import argparse
from math import pi
import pandas as pd
from inverse_kinematics import AnnInverseKinematics, FabrikInverseKinematics

parser = argparse.ArgumentParser(prog='cli')

parser.add_argument('--inverse-kine', required=True, action='store_true',
                        help='use ANN or Fabrik to compute robot inverse kinematics')

parser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                        help='select inverse kinematics engine, Neural Network or Fabrik')

known_args, _ = parser.parse_known_args()
if known_args.method == 'ann':
    parser.add_argument('--model', type=str, required=True,
                        help='select .h5 file with saved mode, required only in ann was choosed')

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

if __name__ == '__main__':
    if ikine_method == 'ann':
        ik_engine = AnnInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)
        ik_engine.load_model(model)
    elif ikine_method == 'fabrik':
        ik_engine = FabrikInverseKinematics(dh_matrix, links_lengths, effector_workspace_limits)

    points_df = pd.read_csv(points_file)
    # convert dataframe to list of points


    angles = [ik_engine.ikine(pos) for pos in points_df]
