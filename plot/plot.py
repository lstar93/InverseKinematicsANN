""" Plotters for robot environment """
#!/usr/bin/env python

import numpy.matlib
import matplotlib.pyplot as plt

class Plotter:
    """ Plotter class """
    limits = []
    rounding = 5

    @staticmethod
    def __round_all_pts(axis, pts):
        """ Round all points to make them more plottable """
        return [round(x[axis], Plotter.rounding) for x in pts]

    @staticmethod
    def __figure(points, size=(5,5)):
        """ Create 3D figure, initialize axes and set labels and colors """
        fig = plt.figure(figsize=size)
        axes = fig.add_subplot(111, projection='3d')
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        colors = [list(x) for x in numpy.random.rand(len(points), 3)]
        return fig, axes, colors

    @staticmethod
    def __set_axes_limits(axes):
        if len(Plotter.limits) == 3:
            axes.set_xlim(*Plotter.limits[0])
            axes.set_ylim(*Plotter.limits[1])
            axes.set_zlim(*Plotter.limits[2])

    @staticmethod
    def set_data_rounding(round_upto):
        """ Set plotted data rounding """
        Plotter.rounding = round_upto

    @staticmethod
    def set_limits(limx, limy, limz):
        """ Set plot limits """
        Plotter.limits = [limx, limy, limz]

    @staticmethod
    def __points_scatter(axes, points, points_colors, path=False):
        """ Simple scatter with previously rounded points """
        rounded_points = [Plotter.__round_all_pts(0, points),
                          Plotter.__round_all_pts(1, points),
                          Plotter.__round_all_pts(2, points)]
        # add scatter plot for points
        axes.scatter(*rounded_points, color=points_colors)

        # optionally add path connecting points
        if path:
            axes.plot(*rounded_points, color=points_colors)

    @staticmethod
    def plot_points_3d(points, path=False, dot_color=None):
        """ Plot poits cloud in 3D space """
        _, axes, colors = Plotter.__figure(points)

        Plotter.__set_axes_limits(axes)

        dot_colors = colors if dot_color is None else [dot_color for _ in colors]

        Plotter.__points_scatter(axes, points, dot_colors, path)

        plt.show()

    @staticmethod
    def plot_joint_points_3d(points_first, points_second, path=False):
        """ Plot poits clouds in 3D space """
        points_0 = list(points_first)
        points_1 = list(points_second)
        _, axes, _ = Plotter.__figure(points_0+points_1)

        Plotter.__set_axes_limits(axes)

        Plotter.__points_scatter(axes, points_0, 'r', path)
        Plotter.__points_scatter(axes, points_1, 'b', path)

        plt.show()

    # matplotlib cannot resize all axes to the same scale so very small numbers make plots hard
    # to read, so all very small numbers will be rounded to 0 for plotting purposes only
    @staticmethod
    def plot_robot(joints, goal_points=None):
        """ Plot robot view """
        # rounding = 10 # set rounding to 10 decimal places for whole plot
        _, axes, _ = Plotter.__figure(joints)

        Plotter.__set_axes_limits(axes)

        # plot robot arms
        points = [pt.to_list() for pt in joints]
        rounded_points = [Plotter.__round_all_pts(0, points),
                          Plotter.__round_all_pts(1, points),
                          Plotter.__round_all_pts(2, points)]
        axes.plot(*rounded_points, color='r')

        # plot robot joints
        Plotter.__points_scatter(axes, points, 'b')

        # optionally plot all robot goal points
        if goal_points is not None:
            Plotter.__points_scatter(axes, goal_points, 'b')

        plt.show()
