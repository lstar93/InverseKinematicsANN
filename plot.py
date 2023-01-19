#!/bin/python3
import numpy.matlib

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import matplotlib.pyplot as plt

# helper metho to generate figure, colors and labels
def figure(points):
    """ Create 3D figure """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(points),3)]
    return fig, ax, colors

def plot_points_3d(points, path=False):
    """ Plot poits cloud in 3D space """
    points = list(points)
    _, axes, colors = figure(points)
    rounded_points = [round(axes[0], 1), round(axes[1], 1), round(axes[2], 1)]

    # add scatter plot for points
    for axe_x, axe_y, axe_z, color in zip(*rounded_points, colors):
        axes.scatter(axe_x, axe_y, axe_z, color=color)

    # optionally add path connecting points
    if path:
        axes.plot(*rounded_points, color='r')

    plt.show()

# matplotlib cannot resize all axes to the same scale so very small numbers make plots impossible to
# thus all very small numbers will be rounded to 0 for plotting purposes only
def plot_robot(joints, points = None):
    """ Plot robot view """
    rounding= 10 # set rounding to 10 decimal places for whole plot
    _, axes, colors = figure(points)

    for joint, color in zip(joints, colors[0:len(joints)]):
        axes.scatter([round(x.x, rounding) for x in joint],
        [round(x.y, rounding) for x in joint],
        [round(x.z, rounding) for x in joint],
        color=color)

        axes.plot3D([round(x.x, rounding) for x in joint],
        [round(x.y, rounding) for x in joint],
        [round(x.z, rounding) for x in joint],
        color=color)

    for point, color in zip(points, colors[len(joints):]):
        axes.scatter(round(point.x, rounding),
        round(point.y, rounding),
        round(point.z, rounding),
        color=color)

    plt.show()
