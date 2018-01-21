import random
import math
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController
from flow.controllers.base_controller import BaseController
import collections
import numpy as np
import joblib


class TrainedPolicyController(BaseController):

    def __init__(self, veh_id, max_deacc=15, tau=0, dt=0.1, pkl_file="",
                 vehicles=None, fail_safe=None):
        """
        Controller that takes in a previously trained policy and uses it to
        generate its actions.

        Parameters
        ----------
        veh_id: str
            unique vehicle identifier
        max_deacc: float, optional
            maximum possible deceleration by the vehicle, in m/s^2
        tau: float, optional
            time delay, in seconds
        dt: float, optional
            size of time steps in the environments
        pkl_file: str
            path to a pkl file containing the controller
        vehicles: Vehicles type
            vehicle class that is used in the environment
        fail_safe: str, optional
            type of fail-safe employed by the vehicle, default is None
        """
        controller_params = {"delay": tau / dt, "max_deaccel": max_deacc,
                             "noise": 0, "fail_safe": fail_safe}

        super().__init__(veh_id, controller_params=controller_params)

        data = joblib.load(pkl_file)
        self.policy = data['policy']
        self.env = data['env'].wrapped_env.env.env.unwrapped
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high

    def get_accel(self, env):
        """
        Assuming that the environment has a method that returns the appropriate
        state for the vehicles of type TrainedPolicyController, it calls that
        method to get an observation, uses it to query the trained policy

        Parameters
        ----------
        env: Sumo environment class
            current state of the action environment the trained policy vehicle
            resides int

        Returns
        -------
        acc: float
            an acceleration
        """
        # update the scenario and the state of vehicles in the trained policy's
        # environment to match the state of vehicles in the actual environment
        self.env.vehicles = env.vehicles
        self.env.scenario = env.scenario

        # update the trained policy's ids and rl_ids lists to match the current
        # environment
        self.env.ids = env.ids
        self.env.rl_ids = [self.veh_id]

        # get the observation and actions from the modified environment
        observation = self.env.get_state()
        actions = self.policy.get_action(observation.T)
        actions = np.clip(actions[0], self.low, self.high)

        # TODO: find more robust way of doing this, maybe by calling the traci
        # TODO: connection from the environment somehow
        # apply the requested actions
        acceleration = actions[0]
        direction = np.array([np.round(actions[1])])

        non_lane_changing_veh = \
            [self.env.timer <= 1.0 + self.env.vehicles.get_state(self.veh_id, 'last_lc')]
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        env.apply_acceleration([self.veh_id], acc=acceleration)
        env.apply_lane_change([self.veh_id], direction=direction)

        return acceleration

    def reset_delay(self, env):
        pass
