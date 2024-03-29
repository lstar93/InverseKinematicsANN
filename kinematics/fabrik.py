""" FABRIK algorithm """
#!/usr/bin/env python

# pylint: disable=W0511 # suppress TODOs

from kinematics.point import Point, get_distance_between, get_point_between


class Fabrik:
    """ FABRIK stands from forward and backward reaching inverse kinematics ->\
            https://www.youtube.com/watch?v=UNoX65PRehA&feature=emb_title
            https://www.academia.edu/9165835/FABR...matics_problem """
    def __init__(self, joints_distances, err_margin = 0.001, max_iter_num = 100):
        self.joints_distances = joints_distances
        self.err_margin = err_margin
        self.max_iter_num = max_iter_num

    # Compute backward iteration
    def __backward(self, points, joints_goal_points):
        """ Calculate backward joint positions -> \
            from goal position to point close to the start joint """
        points_to_ret = [joints_goal_points] # goal point should be the last point in array
        positions = list(reversed(points[:-1]))
        distances = list(reversed(self.joints_distances[:-1]))
        for next_point, distance in zip(positions, distances):
            middle_stage = get_point_between(points_to_ret[-1], next_point, distance)
            # todo: check middle stage boundries
            points_to_ret.append(middle_stage)
        return list(reversed(points_to_ret))

    # Compute forward iteration
    def __forward(self, points, start_point):
        """ Calculate forward joint positions -> \
            from start position to point close to the goal position """
        points_to_ret = [start_point] # start point should be the first point in array
        positions = points[1:]
        distances = self.joints_distances[1:]
        for next_point, distance in zip(positions, distances):
            middle_stage = get_point_between(points_to_ret[-1], next_point, distance)
            # todo: check middle stage boundries
            points_to_ret.append(middle_stage)
        return points_to_ret

    def calculate(self, init_joints_positions, goal_effector_position):
        """ Calculate all joints positions joining backward and forward methods """
        if not all(x == len(init_joints_positions)
                for x in (len(init_joints_positions), len(self.joints_distances))):
            raise ValueError('Input vectors should have equal lengths!')

        current_join_positions = init_joints_positions
        start_point = init_joints_positions[0]
        joints_goal_points = Point(goal_effector_position)
        start_error = 1
        goal_error = 1
        step = 0

        while (((start_error > self.err_margin)
                or (goal_error > self.err_margin))
                    and (self.max_iter_num > step)):
            backward = self.__backward(current_join_positions, joints_goal_points)
            start_error = get_distance_between(backward[0], start_point)
            forward = self.__forward(backward, start_point)
            goal_error = get_distance_between(forward[-1], joints_goal_points)
            current_join_positions = forward
            step += 1

        return current_join_positions
