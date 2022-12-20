#!/bin/python3
import numpy.matlib

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import matplotlib.pyplot as plt

# helper metho to generate figure, colors and labels
def figure(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(points),3)]  
    return fig, ax, colors

def plot_points_3d(points, path=False):
    points = list(points)
    _, ax, colors = figure(points)

    # round points up to some decimal places
    round_pts = lambda ax: [round(x[ax], 1) for x in points]
    rounded_points = [round_pts(0), round_pts(1), round_pts(2)]

    # add scatter plot for points
    for x,y,z,c in zip(*rounded_points, colors):
        ax.scatter(x, y, z, color=c)

    # add path connecting points
    if path:
        ax.plot(*rounded_points, color='r')

    plt.show()

# matplotlib cannot resize all axes to the same scale so very small numbers make plots impossible to  
# thus all very small numbers will be rounded to 0 for plotting purposes only
def plot_robot(joints, points = []):
    PRECISION = 10 # set precision to 10 decimal places for whole plot
    _, ax, colors = figure(points) 
    for j,c in zip(joints,colors[0:len(joints)]):
        ax.scatter([round(x.x, PRECISION) for x in j], [round(x.y, PRECISION) for x in j], [round(x.z, PRECISION) for x in j], color=c)
        ax.plot3D([round(x.x, PRECISION) for x in j], [round(x.y, PRECISION) for x in j], [round(x.z, PRECISION) for x in j], color=c)

    for p,c in zip(points,colors[len(joints):]):
        ax.scatter(round(p.x, PRECISION), round(p.y, PRECISION), round(p.z, PRECISION), color=c)

    plt.show()