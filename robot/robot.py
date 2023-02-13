""" Planar robot """
#!/usr/bin/env python

# pylint: disable=W1401 # string constant might be missing r prefix
# pylint: disable=W0105 # string has no effect
# pylint: disable=C0413 # imports should be placed at the top of the module

# DH notation
"""
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
"""

from math import pi
from dataclasses import dataclass

# 6 DOF robot DH matrix, links lengths, workspace and joints limits
@dataclass
class SixDOFRobot():
    """ 6 DOF robot math description """
    dh_matrix = [[0, pi/2, 0, 0], [2, 0, 0, 0], [0, 2, 2, 2], [pi/2, 0, 0, 0]]
    effector_workspace_limits = {'x': [0,6], 'y': [-6,6], 'z': [-3,6]}
    links_lengths = [2, 2, 2, 2]


class OutOfRobotReachException(Exception):
    """ Robot manipulator exception class """
