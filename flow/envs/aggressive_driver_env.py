'''
Environment for training an aggressive driver in a multilane ring road
'''

from flow.envs.lane_changing import SimpleLaneChangingAccelerationEnvironment

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
import numpy as np


class AggressiveDriverEnvironment(SimpleLaneChangingAccelerationEnvironment):
    """
    Environment used to train an aggressive driver behavior.

    The autonomous vehicle is allowed to move up to 1.75x the speed limit (as
    specified in the run script), and rewards only its speed in the network.

    The autonomous vehicle is able to see its speed, and the speeds and relative
    position of the leading vehicle in its lane the lanes twice to the left and
    to the right of it.
    """
    def __init__(self, env_params, sumo_params, scenario):

        super().__init__(env_params, sumo_params, scenario)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class.
        Encourages high speeds from the rl vehicle only. Also, in order to
        discourage unnecessary lane changes, a small penalty is imposed on
        changing lanes.
        """
        curr_vel = self.vehicles.get_speed(veh_id=self.rl_ids[0])

        total_lane_change_penalty = 0
        for veh_id in self.rl_ids:
            if self.vehicles.get_state(veh_id, "last_lc") == self.timer:
                total_lane_change_penalty -= 1

        return curr_vel + total_lane_change_penalty

    @property
    def observation_space(self):
        """
        See parent class

        An observation consists of the velocity, absolute position, and lane
        index of each vehicle in the fleet
        """
        speed = Box(low=0, high=np.inf, shape=(11,))
        headway = Box(low=0., high=np.inf, shape=(11,))
        return Tuple((speed, headway))

    @property
    def action_space(self):
        """
        See parent class

        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) lane-change action from -1 to 1, used to determine the
           lateral direction the vehicle will take.
        """
        max_deacc = self.env_params.max_deacc
        max_acc = self.env_params.max_acc

        # FIXME hard coded
        lb = [-abs(max_deacc), -1] * 1
        ub = [max_acc, 1] * 1

        return Box(np.array(lb), np.array(ub))

    def get_state(self):
        """
        See parent class

        The state is an array the velocities, absolute positions for the closest
        vehicles in front of the rl cars in the left two lanes, the rl lane,
        and the right two lanes.
        """
        this_vel = self.vehicles.get_speed(self.rl_ids[0])
        this_pos = self.get_x_by_id(self.rl_ids[0])
        this_lane = self.vehicles.get_lane(self.rl_ids[0])

        all_cars = [(veh_id, self.get_x_by_id(veh_id), self.vehicles.get_lane(veh_id))
                    for veh_id in self.ids if veh_id not in self.rl_ids]

        lanes = [[] for _ in range(self.scenario.lanes)]

        for car_tuple in all_cars:
            veh_id, pos, lane = car_tuple
            lanes[lane].append((veh_id, pos))

        obs_headways = [0 for _ in range(5)]
        obs_tailways = [0 for _ in range(5)]
        obs_head_speeds = [0 for _ in range(5)]
        obs_tail_speeds = [0 for _ in range(5)]

        # Find closest vehicle in lane
        for l in range(this_lane - 2, this_lane + 3):
            # If out of bounds
            if l < 0 or l >= self.scenario.lanes:
                continue

            closest_head_dist = self.scenario.length
            closest_head_id = None
            closest_tail_dist = self.scenario.length
            closest_tail_id = None

            for car_id, pos in lanes[l]:
                if (pos - this_pos) % self.scenario.length < closest_head_dist:
                    # This is now the closest leading car in lane l
                    closest_head_dist = (pos - this_pos) % self.scenario.length
                    closest_head_id = car_id

                if (this_pos - pos) % self.scenario.length < closest_tail_dist:
                    # This is now the closest following car in lane l
                    closest_tail_dist = (this_pos - pos) % self.scenario.length
                    closest_tail_id = car_id

            obs_headways[l - (this_lane - 2)] = closest_head_dist
            obs_tailways[l - (this_lane - 2)] = closest_tail_dist
            if closest_head_id is not None:
                obs_head_speeds[l - (this_lane - 2)] = \
                    self.vehicles.get_speed(closest_head_id)
            if closest_tail_id is not None:
                obs_tail_speeds[l - (this_lane - 2)] = \
                    self.vehicles.get_speed(closest_tail_id)

        return np.array([[this_vel] + obs_head_speeds + obs_tail_speeds,
                         [0] + obs_headways + obs_tailways]).T

    def apply_rl_actions(self, actions):
        pass
        """
        See parent class

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands for the duration of the lane
        change and return negative rewards for actions during that lane change.
        if a lane change isn't applied, and sufficient time has passed, issue an
        acceleration like normal.
        """
        acceleration = actions[::2]
        direction = np.round(actions[1::2])

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.rl_ids]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.timer <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)
