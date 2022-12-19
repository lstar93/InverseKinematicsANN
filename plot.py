#!/bin/python3
import numpy.matlib

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import matplotlib.pyplot as plt

def plot_points_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(points),3)] 

    for p,c in zip(points,colors):
        ax.scatter(round(p.x, 1), round(p.y, 5), round(p.z, 5), color=c)

    plt.show()

def plot_list_points_cloud(points_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(points_list),3)] 

    for p,c in zip(points_list,colors):
        ax.scatter(round(p[0], 1), round(p[1], 5), round(p[2], 5), color=c)

    plt.show()

# matplotlib cannot resize all axes to the same scale so very small numbers make plots impossible to analyze 
# thus all very small numbers will be rounded to 0 for plotting purposes only
def plot_robot(joints, points = []):
    PRECISION = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = [list(x) for x in numpy.random.rand(len(joints) + len(points),3)] 
    for j,c in zip(joints,colors[0:len(joints)]):
        ax.scatter([round(x.x, PRECISION) for x in j], [round(x.y, PRECISION) for x in j], [round(x.z, PRECISION) for x in j], color=c)
        ax.plot3D([round(x.x, PRECISION) for x in j], [round(x.y, PRECISION) for x in j], [round(x.z, PRECISION) for x in j], color=c)

    for p,c in zip(points,colors[len(joints):]):
        ax.scatter(round(p.x, PRECISION), round(p.y, PRECISION), round(p.z, PRECISION), color=c)

    plt.show()