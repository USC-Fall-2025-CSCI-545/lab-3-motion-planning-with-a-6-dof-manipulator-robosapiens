#!/usr/bin/env python

import argparse
import time

import adapy
import numpy as np
import rospy


class AdaRRT():
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-3.14, 1.57, 0.33, -3.14, 0, 0])
    joint_upper_limits = np.array([3.14, 5.00, 5.00, 3.14, 3.14, 3.14])

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param state: np.array of new child node's statee
            :returns: child Node object.
            """
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
        """
        :param start_state: Array representing the starting state.
        :param goal_state: Array representing the goal state.
        :param ada: libADA instance.
        :param joint_lower_limits: List of lower bounds of each joint.
        :param joint_upper_limits: List of upper bounds of each joint.
        :param ada_collision_constraint: Collision constraint object.
        :param step_size: Distance between nodes in the RRT.
        :param goal_precision: Maximum distance between RRT and goal before
            declaring completion.
        :param sample_near_goal_prob:
        :param sample_near_goal_range:
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = AdaRRT.Node(start_state, None)
        self.goal = AdaRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdaRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdaRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.max_iter = max_iter

    # def build(self):
    #     """
    #     Build an RRT.

    #     In each step of the RRT:
    #         1. Sample a random point.
    #         2. Find its nearest neighbor.
    #         3. Attempt to create a new node in the direction of sample from its
    #             nearest neighbor.
    #         4. If we have created a new node, check for completion.

    #     Once the RRT is complete, add the goal node to the RRT and build a path
    #     from start to goal.

    #     :returns: A list of states that create a path from start to
    #         goal on success. On failure, returns None.
    #     """
    #     for _ in range(self.max_iter):
    #         try:
    #             point = self._get_random_sample()
    #             ngh = self._get_nearest_neighbor(point)
    #             extended = self._extend_sample(point,ngh)
    #             if extended and self._check_for_completion(extended):
    #                 self.goal.parent = extended
    #                 return self._trace_path_from_start(self.goal)
    #         except Exception as e:
    #             print("something went wrong: ", e)
    #             return
            
    def build(self):
        """
        Build an RRT.

        In each step of the RRT:
            1. Sample a random point (with 20% probability near goal, 80% uniform).
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for _ in range(self.max_iter):
            try:
                # Sample near goal with 20% probability, otherwise sample uniformly
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
                print("something went wrong: ", e)
                return
            
    def _get_random_sample_near_goal(self):
        """
        Samples a point near the goal within a distance of 0.05 along each axis.

        :returns: A vector representing a randomly sampled point near the goal.
        """
        # Generate random offsets in range [-0.05, 0.05] for each dimension
        offsets = np.random.uniform(-0.05, 0.05, size=self.goal.state.shape)
        sample = self.goal.state + offsets
        
        # Clamp to joint limits to ensure validity
        sample = np.clip(sample, self.joint_lower_limits, self.joint_upper_limits)
        
        return sample

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """

        return np.random.uniform(self.joint_lower_limits,self.joint_upper_limits)

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        mn = None
        curr_mn = float('inf')
        for neighbor in self.start:
            dist = np.linalg.norm(neighbor.state - sample)  # Fixed: added .norm() and computed distance
            if dist < curr_mn:
                curr_mn = dist
                mn = neighbor  # Fixed: storing the node, not just state
        return mn
        # FILL in your code here

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        direction = sample - neighbor.state  # Fixed: sample - neighbor (not neighbor - sample)
        distance = np.linalg.norm(direction)  # Added: compute distance
        
        if distance == 0:  # Added: handle edge case
            return None
        
        direction = direction / distance  # Added: normalize direction
        new_config = neighbor.state + direction * self.step_size  # Fixed: start from neighbor, not sample
        
        # Clamp to joint limits
        new_config = np.clip(new_config, self.joint_lower_limits, self.joint_upper_limits)  # Cleaner way
        
        if self._check_for_collision(new_config):
            return None
        return neighbor.add_child(new_config)

        
    def _check_for_completion(self, node):
        """
        Check whether node is within self.goal_precision distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
       
        return np.linalg.norm(node.state - self.goal.state) <= self.goal_precision


    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        if node is None:
            node = self.goal
        path = []
        while node is not None:  
            path.append(node.state)  
            node = node.parent
        path.reverse()
        return path

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision.
        """
        if self.ada_collision_constraint is None:
            return False
        return self.ada_collision_constraint.is_satisfied(
            self.ada.get_arm_state_space(),
            self.ada.get_arm_skeleton(), sample)


def main(is_sim):
    
    if not is_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('adarrt', [])

    # instantiate an ada
    ada = adapy.Ada(is_sim)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-1.72, 4.44, 2.02, -2.04, 2.66, 1.39]
    delta = 0.25
    eps = 0.2

    if is_sim:
        ada.set_positions(goalConfig)
    else:
        input("Please move arm to home position with the joystick. " +
            "Press ENTER to continue...")


    # launch viewer
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add objects to world
    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.7, 0.707107, 0, 0, 0.707107]
    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    table = world.add_body_from_urdf(tableURDFUri, tablePose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            [can, table])
    full_collision_constraint = ada.get_full_collision_constraint(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            collision_free_constraint)

    # easy goal
    adaRRT = AdaRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps)

    rospy.sleep(1.0)

    if not is_sim:
        ada.start_trajectory_executor()

    path = adaRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))
        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((0.0 + i, waypoint))

        t0 = time.perf_counter()
        traj = ada.compute_joint_space_path(
            ada.get_arm_state_space(), waypoints)
        t = time.perf_counter() - t0
        print(str(t) + "seconds elapsed")
        input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', dest='is_sim', action='store_true')
    parser.add_argument('--real', dest='is_sim', action='store_false')
    parser.set_defaults(is_sim=True)
    args = parser.parse_args()
    main(args.is_sim)