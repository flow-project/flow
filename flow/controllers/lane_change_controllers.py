"""Contains a list of custom lane change controllers."""

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
        # get current acceleration controller
        acc_controller = env.k.vehicle.get_acc_controller(self.veh_id)

        # get lane leaders and followers
        lane_leaders = env.k.vehicle.get_lane_leaders(self.veh_id)
        lane_followers = env.k.vehicle.get_lane_followers(self.veh_id)

        # get current lane number
        current_lane = env.k.vehicle.get_lane(self.veh_id)

        # get left and right leader and follower
        left_leader = None
        right_leader = None
        left_follower = None
        right_follower = None
        for veh in lane_leaders:
            if env.k.vehicle.get_lane(veh) == current_lane - 1:
                left_leader = veh
            elif env.k.vehicle.get_lane(veh) == current_lane + 1:
                right_leader = veh

        for veh in lane_followers:
            if env.k.vehicle.get_lane(veh) == current_lane - 1:
                left_follower = veh
            elif env.k.vehicle.get_lane(veh) == current_lane + 1:
                right_follower = veh

        # acceleration if the ego vehicle remains in current lane.
        ego_accel_controller = env.k.vehicle.get_acc_controller(self.veh_id)
        acc_in_present_lane = ego_accel_controller.get_accel(env)

        # assert to make sure the CFM have the get_custom_accel()  # TODO

        # acceleration if the ego vehicle is in the lane to the left
        acc_in_left_lane = ego_accel_controller.get_accel(env)  # FIXME

        # acceleration if the ego vehicle is in the lane to the right
        acc_in_right_lane = ego_accel_controller.get_accel(env)  # FIXME

        # acceleration of the new follower if left lane change is made
        l_f_accel_controller = env.k.vehicle.get_acc_controller(left_follower)
        left_lane_follower_acc = l_f_accel_controller.get_accel(env)  # FIXME

        # acceleration of the new follower if right lane change is made
        r_f_accel_controller = env.k.vehicle.get_acc_controller(right_follower)
        right_lane_follower_acc = r_f_accel_controller.get_accel(env)  # FIXME

        # determine lane change action
        if acc_in_left_lane >= - self.left_beta and \
                left_lane_follower_acc >= -self.left_beta and \
                acc_in_left_lane >= acc_in_present_lane + self.left_delta:
            action = 1
        elif acc_in_right_lane >= - self.right_beta and \
                right_lane_follower_acc >= -self.right_beta and \
                acc_in_right_lane >= acc_in_present_lane + self.right_delta:
            action = -1
        else:
            action = 0

        return action
