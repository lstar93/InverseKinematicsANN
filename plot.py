""" Plotters for robot environment """
#!/usr/bin/env python

import numpy.matlib
import matplotlib.pyplot as plt

def round_all_pts(axis, pts, upto = 2):
    """ Rout all points to make them more plottable """
    return [round(x[axis], upto) for x in pts]

# helper metho to generate figure, colors and labels
def figure(points, size=(5,5)):
    """ Create 3D figure """
    fig = plt.figure(figsize=size)
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(points), 3)]
    return fig, axes, colors

def plot_points_3d(points, path=False, dot_color = ''):
    """ Plot poits cloud in 3D space """
    _, axes, colors = figure(points)
    if dot_color != '':
        colors = [dot_color for _ in colors]

    rounded_points = [round_all_pts(0, points), round_all_pts(1, points), round_all_pts(2, points)]

    # add scatter plot for points
    for axe_x, axe_y, axe_z, color in zip(*rounded_points, colors):
        axes.scatter(axe_x, axe_y, axe_z, color=color)

    # optionally add path connecting points
    if path:
        axes.plot(*rounded_points, color=dot_color)

    plt.show()

def plot_joint_points_3d(points_first, points_sec, path=False):
    """ Plot poits clouds in 3D space """
    points_0 = list(points_first)
    points_1 = list(points_sec)
    _, axes, colors = figure(points_0+points_1)

    def round_pts(axis, pts):
        return [round(x[axis], 1) for x in pts]

    rounded_points_0 = [round_pts(0, points_0), round_pts(1, points_0), round_pts(2, points_0)]
    rounded_points_1 = [round_pts(0, points_1), round_pts(1, points_1), round_pts(2, points_1)]

    # add scatter plot for points
    for axe_x, axe_y, axe_z, _ in zip(*rounded_points_0, colors):
        axes.scatter(axe_x, axe_y, axe_z, color='r')

    for axe_x, axe_y, axe_z, _ in zip(*rounded_points_1, colors):
        axes.scatter(axe_x, axe_y, axe_z, color='b')

    # optionally add path connecting points
    if path:
        axes.plot(*rounded_points_0, color='r')
        axes.plot(*rounded_points_1, color='b')

    plt.show()

# matplotlib cannot resize all axes to the same scale so very small numbers make plots hard
# to read, so all very small numbers will be rounded to 0 for plotting purposes only
def plot_robot(joints, goal_points = None):
    """ Plot robot view """
    # rounding = 10 # set rounding to 10 decimal places for whole plot
    _, axes, colors = figure(joints)

    points = [pt.to_list() for pt in joints]
    rounded_points = [round_all_pts(0, points), round_all_pts(1, points), round_all_pts(2, points)]

    axes.set_xlim(-3,3)
    axes.set_ylim(-3,3)
    axes.set_zlim(-3,3)

    # add scatter plot for points
    for axe_x, axe_y, axe_z, color in zip(*rounded_points, colors):
        axes.scatter(axe_x, axe_y, axe_z, color=color)

    axes.plot(*rounded_points, color='r')

    if goal_points is not None:
        rounded_goal_points = [round_all_pts(0, goal_points),
                               round_all_pts(1, goal_points),
                               round_all_pts(2, goal_points)]
        axes.scatter(*rounded_goal_points, color='b')

    plt.show()
