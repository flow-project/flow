"""Contains a list of custom lane change controllers."""
import numpy as np
from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SumoLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle."""

    def __init__(self, veh_id):
        super().__init__(veh_id, lane_change_params={})
        self.SumoController = True

    def get_lane_change_action(self, env):
        """See parent class."""
        return None

class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane."""

    def get_lane_change_action(self, env):
        """See parent class."""
        return 0


class SafeAggressiveLaneChanger(BaseLaneChangeController):
    def __init__(self, veh_id, target_velocity, threshold=0.75):
        """
        A lane-changing model used to perpetually keep a vehicle in the same
        lane.
        Attributes
        ----------
        veh_id: str
            unique vehicle identifier
        """
        super().__init__(veh_id)
        self.veh_id = veh_id
        self.threshold_velocity = min(max(np.random.normal(threshold/2.0), 0), threshold) * target_velocity

    def get_action(self, env):
        if env.vehicles.get_speed(self.veh_id) < self.threshold_velocity:
            lane_headways = env.vehicles.get_lane_headways(self.veh_id)
            lane_tailways = env.vehicles.get_lane_tailways(self.veh_id)
            if (len(lane_headways) == 0):
                return 0
            curr_lane = env.vehicles.get_lane(self.veh_id)

            # available_lanes = list(range(max(curr_lane-1,0), min(curr_lane + 1, env.scenario.lanes) +1))
            available_headways = lane_headways[max(curr_lane-1,0): min(curr_lane + 1, env.scenario.lanes) +1]
            desired_available_lane = np.argmax(available_headways)
            # desired_lane = available_lanes[desired_available_lane]
            if lane_tailways[desired_available_lane] < 8:
                return 0
            else:
                return desired_available_lane - 1
        else:
            return 0