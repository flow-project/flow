from flow.envs.base_env import SumoEnvironment
from flow.core import rewards
from flow.controllers.car_following_models import *

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
import numpy as np
from numpy.random import normal


class SimpleLaneChangingAccelerationEnvironment(SumoEnvironment):
    """
    Fully functional environment for multi lane closed loop settings. Takes in
    an *acceleration* and *lane-change* as an action. Reward function is
    negative norm of the difference between the velocities of each vehicle, and
    the target velocity. State function is a vector of the velocities and
    absolute positions for each vehicle.
    """

    @property
    def action_space(self):
        """
        See parent class

        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) lane-change action from -1 to 1, used to determine the
           lateral direction the vehicle will take.
        """
        max_deacc = self.env_params.get_additional_param("max-deacc")
        max_acc = self.env_params.get_additional_param("max-acc")

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
        lane = Box(low=0, high=self.scenario.lanes-1, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos, lane))

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class

        The reward function is negative norm of the difference between the
        velocities of each vehicle, and the target velocity. Also, a small
        penalty is added for rl lane changes in order to encourage mimizing
        lane-changing action.
        """
        # compute the system-level performance of vehicles from a velocity
        # perspective
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes
        for veh_id in self.rl_ids:
            if self.vehicles.get_state(veh_id, "last_lc") == self.timer:
                reward -= 1

        return reward

    def get_state(self):
        """
        See parent class

        The state is an array the velocities, absolute positions, and lane
        numbers for each vehicle.
        """
        return np.array([[self.vehicles.get_speed(veh_id),
                          self.vehicles.get_absolute_position(veh_id),
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

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.timer <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)


class LaneChangeOnlyEnvironment(SimpleLaneChangingAccelerationEnvironment):
    """
    Am extension of SimpleLaneChangingAccelerationEnvironment. Autonomous
    vehicles in this environment can only make lane-changing decisions. Their
    accelerations, on the other hand, are controlled by an human car-following
    model specified under "rl_acc_controller" in the in additional_params
    attribute of env_params.
    """

    def __init__(self, env_params, sumo_params, scenario):

        super().__init__(env_params, sumo_params, scenario)

        # acceleration controller used for rl cars
        self.rl_controller = dict()

        for veh_id in self.rl_ids:
            acc_params = env_params.get_additional_param("rl_acc_controller")
            self.rl_controller[veh_id] = \
                acc_params[0](veh_id=veh_id, **acc_params[1])

    @property
    def action_space(self):
        """
        See parent class
        
        Actions are: a continuous direction for each rl vehicle
        """
        return Box(low=-1, high=1, shape=(self.vehicles.num_rl_vehicles,))

    def apply_rl_actions(self, actions):
        """
        see parent class

        Actions are applied to rl vehicles as follows:
        - accelerations are derived using the user-specified accel controller
        - lane-change commands are collected from rllab
        """
        direction = actions

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = \
            [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.timer <= self.lane_change_duration + self.vehicles[veh_id]['last_lc']
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_lane_change(sorted_rl_ids, direction=direction)

        # collect the accelerations for the rl vehicles as specified by the
        # human controller
        acceleration = []
        for veh_id in sorted_rl_ids:
            acceleration.append(self.rl_controller[veh_id].get_action(self))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
