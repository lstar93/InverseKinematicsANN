""" Simple CLI to use Fabrik or ANN ikine methods """
#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(prog='cli')

parser.add_argument('--inverse-kine', required=True, action='store_true',
                        help='use ANN or Fabrik to compute robot inverse kinematics')

parser.add_argument('--method', required=True, type=str, choices=['ann', 'fabrik'],
                        help='select inverse kinematics engine, Neural Network or Fabrik')

known_args, _ = parser.parse_known_args()
if known_args.method == 'ann':
    parser.add_argument('--model', type=str, required=False,
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
