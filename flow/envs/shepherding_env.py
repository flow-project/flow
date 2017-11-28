
from flow.envs.loop_accel import SimpleAccelerationEnvironment

import numpy as np
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple


class ShepherdingEnv(SimpleAccelerationEnvironment):

    def compute_reward(self, state, rl_actions, **kwargs):
        desired_vel = np.array([self.env_params.additional_params["target_velocity"]] * self.vehicles.num_vehicles)
        curr_vel = np.array(self.vehicles.get_speed())
        diff_vel = np.linalg.norm(desired_vel - curr_vel)
        accel = self.vehicles.get_accel(veh_id="all")
        deaccel =  np.linalg.norm([min(0, x) for x in accel])
        return -(0.5 * diff_vel + 0.5 * deaccel)

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

        lb = [-abs(max_deacc), -1] * self.vehicles.num_rl_vehicles
        ub = [max_acc, 1] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class

        An observation consists of the velocity, absolute position, and lane
        index of each vehicle in the fleet
        """
        speed = Box(low=-np.inf, high=np.inf, shape=(self.vehicles.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes - 1, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos, lane))

    def get_state(self):
        """
        See parent class

        The state is an array the velocities, absolute positions, and lane
        numbers for each vehicle.
        """
        return np.array([[self.vehicles.get_speed(veh_id),
                          self.get_x_by_id(veh_id),
                          self.vehicles.get_lane(veh_id)]
                         for veh_id in self.sorted_ids])

    def apply_rl_actions(self, actions):
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
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        # sorted_rl_ids = self.rl_ids

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.timer <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)