"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base_env import Env

from gym.spaces.box import Box

import numpy as np
import random
from scipy.optimize import fsolve

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


class WaveAttenuationEnv(Env):
    """Fully observable wave attenuation environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in a variable density ring road.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        return Box(
            low=0,
            high=1,
            shape=(2 * self.vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.vehicles.get_rl_ids()
        ]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.vehicles.get_speed(veh_id)
            for veh_id in self.vehicles.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 8  # 0.25
        rl_actions = np.array(rl_actions)
        accel_threshold = 0

        if np.mean(np.abs(rl_actions)) > accel_threshold:
            reward += eta * (accel_threshold - np.mean(np.abs(rl_actions)))

        return float(reward)

    def get_state(self):
        """See class definition."""
        speed = [self.vehicles.get_speed(veh_id) / self.scenario.max_speed
                 for veh_id in self.sorted_ids]
        pos = [self.get_x_by_id(veh_id) / self.scenario.length
               for veh_id in self.sorted_ids]

        return np.array(speed + pos)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)

    def reset(self):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # update the scenario
        initial_config = InitialConfig(bunching=50, min_gap=0)
        additional_net_params = {
            'length':
            random.randint(
                self.env_params.additional_params['ring_length'][0],
                self.env_params.additional_params['ring_length'][1]),
            'lanes':
            self.scenario.lanes,
            'speed_limit':
            30,
            'resolution':
            40
        }
        net_params = NetParams(additional_params=additional_net_params)

        self.scenario = self.scenario.__class__(
            self.scenario.orig_name, self.scenario.vehicles,
            net_params, initial_config)

        # solve for the velocity upper bound of the ring
        def v_eq_max_function(v):
            num_veh = self.vehicles.num_vehicles - 1
            # maximum gap in the presence of one rl vehicle
            s_eq_max = (self.scenario.length -
                        self.vehicles.num_vehicles * 5) / num_veh

            v0 = 30
            s0 = 2
            T = 1
            gamma = 4

            error = s_eq_max - (s0 + v * T) * (1 - (v / v0)**gamma)**-0.5

            return error

        v_guess = 4.
        v_eq_max = fsolve(v_eq_max_function, v_guess)[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('v_max:', v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation


class WaveAttenuationPOEnv(WaveAttenuationEnv):
    """POMDP version of WaveAttenuationEnv.

    Note that this environment only works when there is one autonomous vehicle
    on the network.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on

    States
        The state consists of the speed and headway of the ego vehicle, as well
        as the difference in speed between the ego vehicle and its leader.
        There is no assumption on the number of vehicles in the network.

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(3, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        rl_id = self.vehicles.get_rl_ids()[0]
        lead_id = self.vehicles.get_leader(rl_id) or rl_id

        # normalizers
        max_speed = 15.
        max_length = self.env_params.additional_params['ring_length'][1]

        observation = np.array([
            self.vehicles.get_speed(rl_id) / max_speed,
            (self.vehicles.get_speed(lead_id) - self.vehicles.get_speed(rl_id))
            / max_speed,
            self.vehicles.get_headway(rl_id) / max_length
        ])

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        rl_id = self.vehicles.get_rl_ids()[0]
        lead_id = self.vehicles.get_leader(rl_id) or rl_id
        self.vehicles.set_observed(lead_id)
