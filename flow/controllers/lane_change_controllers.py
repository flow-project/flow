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


class SafeAggressiveLaneChanger(BaseLaneChangeController):
    def __init__(self, veh_id, target_velocity, threshold=0.75, desired_lc_time_headway=2.0):
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
        # min(max(np.random.normal(loc=0, scale=threshold/2.0)), threshold) * target_velocity
        self.threshold_velocity = target_velocity * threshold
        self.desired_lc_time_headway = desired_lc_time_headway

    def get_action(self, env):
        if env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            # don't change lange in a junction
            return 0
        curr_speed = env.k.vehicle.get_speed(self.veh_id)
        if curr_speed < self.threshold_velocity:
            lane_headways = env.k.vehicle.get_lane_headways(self.veh_id)
            lane_tailways = env.k.vehicle.get_lane_tailways(self.veh_id)
            if (len(lane_headways) == 0):
                return 0
            elif (len(lane_headways) == 6):
                # special handling for on ramp edge, never switch to offramp lane.
                lane_headways[0] = -1

            curr_lane = env.k.vehicle.get_lane(self.veh_id)
            available_headways = lane_headways[max(curr_lane-1, 0): curr_lane + 2]
            available_tailways = lane_tailways[max(curr_lane-1, 0): curr_lane + 2]
            desired_available_lane = np.argmax(available_headways)

            desired_lane = desired_available_lane + max(curr_lane-1, 0)
            target_follower = env.k.vehicle.get_lane_followers(self.veh_id)[desired_lane]

            if available_tailways[desired_available_lane] < \
                    max(8, env.k.vehicle.get_speed(target_follower) * self.desired_lc_time_headway):
                return 0
            else:
                if curr_lane == 0:
                    return desired_available_lane
                else:
                    return desired_available_lane - 1
        else:
            return 0
