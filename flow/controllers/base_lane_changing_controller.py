class BaseLaneChangingController:

    def __init__(self, veh_id, lane_change_params):
        """
        Base class for lane-changing controllers.

        Instantiates a controller and forces the user to pass a
        lane_changing duration to the controller. Provides the method
        get_safe_lane_change_action to ensure that lane-changes do
        not cause crashes.

        Attributes
        ----------
        veh_id: string
            ID of the vehicle this controller is used for
        lane_change_params: dict
            Dictionary of lane changes params that may optional contain
            "min_gap", which denotes the minimize safe gap (in meters) a car
            is willing to lane-change into.
        """
        self.veh_id = veh_id
        self.lane_change_params = lane_change_params

        # min_gap defines the minimum gap (in meters) that a car is willing
        # to accept in front and behind it to enter a gap
        if "min_gap" in lane_change_params:
            self.min_gap = lane_change_params["min_gap"]
        else:
            self.min_gap = 0.1

    def get_safe_lane_change_action(self, env, target_lane):
        """
        Determines whether a collision will occur if a vehicle enters the target
        lane.

        :param env:
        :param target_lane:
        :return: the safe target lane (requested target lane if action is safe,
                 current lane if the action is not)
        """
        current_lane = env.vehicles[self.veh_id]['lane']

        # if no lane change is being performed, there is no need to check for
        # safety
        if current_lane == target_lane:
            return target_lane

        # if the target lane is not within the range of lanes available, return
        # the current lane
        if target_lane < 0 or target_lane > env.scenario.lanes - 1:
            return current_lane

        lead_id = env.get_leading_car(self.veh_id, target_lane)
        trail_id = env.get_trailing_car(self.veh_id, target_lane)

        # if there is only one vehicle in the environment, or there are no
        # vehicles in the target
        # lane, then lane changing to the target lane is safe
        if (lead_id is None) or (len(env.vehicles) == 1):
            return target_lane

        lead_pos = env.get_x_by_id(lead_id)
        lead_length = env.vehicles[lead_id]['length']

        trail_pos = env.get_x_by_id(trail_id)
        trail_vel = env.vehicles[trail_id]['speed']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']
        this_length = env.vehicles[self.veh_id]['length']

        lead_gap = (lead_pos - this_pos) % env.scenario.length - lead_length
        trail_gap = (this_pos - trail_pos) % env.scenario.length - this_length

        time_step = env.sim_step

        max_acc = env.env_params.max_acc
        max_trail_vel = trail_vel + max_acc * time_step
        max_this_vel = this_vel + max_acc * time_step

        # if vehicle may collide into a vehicle in the target lane in the lane
        # change is performed, opt out of lane change
        if lead_gap - max_this_vel * time_step < self.min_gap or \
                trail_gap - max_trail_vel * time_step < self.min_gap:
            return current_lane
        else:
            return target_lane
