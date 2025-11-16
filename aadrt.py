#!/usr/bin/env python2

import argparse
import time

import adapy
import numpy as np
import rospy


class AdaRRT(object):
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-3.14, 1.57, 0.33, -3.14, 0, 0])
    joint_upper_limits = np.array([3.14, 5.00, 5.00, 3.14, 3.14, 3.14])

    class Node(object):
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            child = AdaRRT.Node(state=state, parent=self)
            self.children.append(child)
            return child

    def __init__(self,
                 start_state,
                 goal_state,
                 ada,
                 joint_lower_limits=None,
                 joint_upper_limits=None,
                 ada_collision_constraint=None,
                 step_size=0.25,
                 goal_precision=0.2,
                 max_iter=10000):

        self.start = AdaRRT.Node(start_state, None)
        self.goal = AdaRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdaRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdaRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.max_iter = max_iter

    def build(self):
        for _ in range(self.max_iter):
            try:
                if np.random.random() < 0.2:
                    point = self._get_random_sample_near_goal()
                else:
                    point = self._get_random_sample()

                ngh = self._get_nearest_neighbor(point)
                extended = self._extend_sample(point, ngh)
                if extended and self._check_for_completion(extended):
                    self.goal.parent = extended
                    return self._trace_path_from_start(self.goal)

            except Exception as e:
                print("something went wrong: {}".format(e))
                return

    def _get_random_sample_near_goal(self):
        offsets = np.random.uniform(-0.05, 0.05, size=self.goal.state.shape)
        sample = self.goal.state + offsets
        sample = np.clip(sample, self.joint_lower_limits, self.joint_upper_limits)
        return sample

    def _get_random_sample(self):
        return np.random.uniform(self.joint_lower_limits, self.joint_upper_limits)

    def _get_nearest_neighbor(self, sample):
        mn = None
        curr_mn = float('inf')
        for neighbor in self.start:
            dist = np.linalg.norm(neighbor.state - sample)
            if dist < curr_mn:
                curr_mn = dist
                mn = neighbor
        return mn

    def _extend_sample(self, sample, neighbor):
        direction = sample - neighbor.state
        distance = np.linalg.norm(direction)

        if distance == 0:
            return None

        direction = direction / distance
        new_config = neighbor.state + direction * self.step_size
        new_config = np.clip(new_config, self.joint_lower_limits, self.joint_upper_limits)

        if self._check_for_collision(new_config):
            return None

        return neighbor.add_child(new_config)

    def _check_for_completion(self, node):
        return np.linalg.norm(node.state - self.goal.state) <= self.goal_precision

    def _trace_path_from_start(self, node=None):
        if node is None:
            node = self.goal
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def _check_for_collision(self, sample):
        if self.ada_collision_constraint is None:
            return False
        return self.ada_collision_constraint.is_satisfied(
            self.ada.get_arm_state_space(),
            self.ada.get_arm_skeleton(),
            sample)


def main(is_sim):

    if not is_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('adarrt', [])

    ada = adapy.Ada(is_sim)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-1.72, 4.44, 2.02, -2.04, 2.66, 1.39]
    delta = 0.25
    eps = 0.2

    if is_sim:
        ada.set_positions(goalConfig)
    else:
        raw_input("Please move arm to home position with the joystick. Press ENTER to continue...")

    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.7, 0.707107, 0, 0, 0.707107]

    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    table = world.add_body_from_urdf(tableURDFUri, tablePose)

    collision_free_constraint = ada.set_up_collision_detection(
        ada.get_arm_state_space(),
        ada.get_arm_skeleton(),
        [can, table]
    )

    full_collision_constraint = ada.get_full_collision_constraint(
        ada.get_arm_state_space(),
        ada.get_arm_skeleton(),
        collision_free_constraint
    )

    adaRRT = AdaRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps
    )

    rospy.sleep(1.0)

    if not is_sim:
        ada.start_trajectory_executor()

    path = adaRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))

        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((float(i), waypoint))

        t0 = time.time()
        traj = ada.compute_joint_space_path(
            ada.get_arm_state_space(), waypoints)
        t = time.time() - t0
        print(str(t) + " seconds elapsed")

        raw_input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', dest='is_sim', action='store_true')
    parser.add_argument('--real', dest='is_sim', action='store_false')
    parser.set_defaults(is_sim=True)
    args = parser.parse_args()
    main(args.is_sim)
