"""Contains a list of custom lane change controllers."""

import sys

from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SimLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return None


class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return 0


class AILaneChangeController(BaseLaneChangeController):
    """A lane-changing controller based on acceleration incentive model.

    Usage
    -----
    See base class for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO/Aimsun identification
    lane_change_params : flow.core.param.SumoLaneChangeParams
        see parent class
    left_delta : float
        used for the incentive criterion for left lane change (default: 2.6)
    right_delta : float
        used for the incentive criterion for right lane change (default: 2.7)
    left_beta : float
        used for the incentive criterion for left lane change (default: 2.6)
    right_beta : float
        used for the incentive criterion for right lane change (default: 2.7)
    """

    def __init__(self,
                 veh_id,
                 lane_change_params=None,
                 left_delta=2.6,
                 right_delta=2.7,
                 left_beta=2.6,
                 right_beta=2.7):
        """Instantiate an AI lane-change controller."""
        BaseLaneChangeController.__init__(
            self,
            veh_id,
            lane_change_params,
        )

        self.veh_id = veh_id
        self.left_delta = left_delta
        self.right_delta = right_delta
        self.left_beta = left_beta
        self.right_beta = right_beta

    def get_lane_change_action(self, env):
        """See parent class."""
        # acceleration if the ego vehicle remains in current lane.
        ego_accel_controller = env.k.vehicle.get_acc_controller(self.veh_id)
        acc_in_present_lane = ego_accel_controller.get_accel(env)

        # get ego vehicle lane number, and velocity
        ego_lane = env.k.vehicle.get_lane(self.veh_id)
        ego_vel = env.k.vehicle.get_speed(self.veh_id)

        # get lane leaders, followers, headways, and tailways
        lane_leaders = env.k.vehicle.get_lane_leaders(self.veh_id)
        lane_followers = env.k.vehicle.get_lane_followers(self.veh_id)
        lane_headways = env.k.vehicle.get_lane_headways(self.veh_id)
        lane_tailways = env.k.vehicle.get_lane_tailways(self.veh_id)

        # determine left and right lane number
        this_edge = env.k.vehicle.get_edge(self.veh_id)
        num_lanes = env.k.network.num_lanes(this_edge)
        l_lane = ego_lane - 1 if ego_lane > 0 else None
        r_lane = ego_lane + 1 if ego_lane < num_lanes - 1 else None

        # compute ego and new follower accelerations if moving to left lane
        if l_lane is not None:
            # get left leader and follower vehicle ID
            l_l = lane_leaders[l_lane]
            l_f = lane_followers[l_lane]

            # ego acceleration if the ego vehicle is in the lane to the left
            if l_l not in ['', None]:
                # left leader velocity and headway
                l_l_vel = env.k.vehicle.get_speed(l_l)
                l_l_headway = lane_headways[l_lane]

                # assert to make sure the CFM have the get_custom_accel()
                try:
                    acc_in_left_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=l_l_vel,
                        h=l_l_headway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)
            else:  # if left lane exists but left leader does not exist
                # in this case we assign None to the leader velocity and
                # large number to headway
                l_l_vel = None
                l_l_headway = 1000
                try:
                    acc_in_left_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=l_l_vel,
                        h=l_l_headway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)

            # follower acceleration if the ego vehicle is in the left lane
            if l_f not in ['', None]:
                # left follower velocity and headway
                l_f_vel = env.k.vehicle.get_speed(l_f)
                l_f_tailway = lane_tailways[l_lane]

                l_f_accel_controller = env.k.vehicle.get_acc_controller(l_f)
                try:
                    left_lane_follower_acc = l_f_accel_controller. \
                        get_custom_accel(
                         this_vel=l_f_vel,
                         lead_vel=ego_vel,
                         h=l_f_tailway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)
            else:  # if left lane exists but left follower does not exist
                # in this case we assign maximum acceleration
                left_lane_follower_acc = ego_accel_controller.max_accel
        else:
            acc_in_left_lane = None
            left_lane_follower_acc = None

        # compute ego and new follower accelerations if moving to right lane
        if r_lane is not None:
            # get right leader and follower vehicle ID
            r_l = lane_leaders[r_lane]
            r_f = lane_followers[r_lane]

            # ego acceleration if the ego vehicle is in the lane to the right
            if r_l not in ['', None]:
                # right leader velocity and headway
                r_l_vel = env.k.vehicle.get_speed(r_l)
                r_l_headway = lane_headways[r_lane]

                try:
                    acc_in_right_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=r_l_vel,
                        h=r_l_headway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)
            else:  # if right lane exists but right leader does not exist
                # in this case we assign None to the leader velocity and
                # large number to headway
                r_l_vel = None
                r_l_headway = 1000
                try:
                    acc_in_right_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=r_l_vel,
                        h=r_l_headway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)

            # follower acceleration if the ego vehicle is in the right lane
            if r_f not in ['', None]:
                # right follower velocity and headway
                r_f_vel = env.k.vehicle.get_speed(r_f)
                r_f_headway = lane_tailways[r_lane]

                r_f_accel_controller = env.k.vehicle.get_acc_controller(r_f)
                try:
                    right_lane_follower_acc = r_f_accel_controller.\
                        get_custom_accel(
                         this_vel=r_f_vel,
                         lead_vel=ego_vel,
                         h=r_f_headway)
                except NotImplementedError:
                    print(
                        "====================================================\n"
                        "The get_custom_accel() method is not implemented for\n"
                        "the selected Car Following model. Please implement  \n"
                        " the method or use another Car Following model      \n"
                        "=====================================================")
                    sys.exit(1)
            else:  # if right lane exists but right follower does not exist
                # assign maximum acceleration
                right_lane_follower_acc = ego_accel_controller.max_accel
        else:
            acc_in_right_lane = None
            right_lane_follower_acc = None

        # determine lane change action
        if l_lane is not None and acc_in_left_lane >= - self.left_beta and \
                left_lane_follower_acc >= -self.left_beta and \
                acc_in_left_lane >= acc_in_present_lane + self.left_delta:
            action = 1
        elif r_lane is not None and acc_in_right_lane >= - self.right_beta and \
                right_lane_follower_acc >= -self.right_beta and \
                acc_in_right_lane >= acc_in_present_lane + self.right_delta:
            action = -1
        else:
            action = 0

        return action
