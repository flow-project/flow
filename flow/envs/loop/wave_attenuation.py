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

from copy import deepcopy
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


def v_eq_max_function(v, *args):
    """Return the error between the desired and actual equivalent gap."""
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

    return error


class WaveAttenuationEnv(Env):
    """Fully observable wave attenuation environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in a variable density ring road.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original scenario definition.

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

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        mean_actions = np.mean(np.abs(np.array(rl_actions)))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            reward += eta * (accel_threshold - mean_actions)

        return float(reward)

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
               for veh_id in self.k.vehicle.get_ids()]

        return np.array(speed + pos)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # reset the step counter
        self.step_counter = 0

        # update the scenario
        initial_config = InitialConfig(bunching=50, min_gap=0)
        length = random.randint(
            self.env_params.additional_params['ring_length'][0],
            self.env_params.additional_params['ring_length'][1])
        additional_net_params = {
            'length':
                length,
            'lanes':
                self.net_params.additional_params['lanes'],
            'speed_limit':
                self.net_params.additional_params['speed_limit'],
            'resolution':
                self.net_params.additional_params['resolution']
        }
        net_params = NetParams(additional_params=additional_net_params)

        self.scenario = self.scenario.__class__(
            self.scenario.orig_name, self.scenario.vehicles,
            net_params, initial_config)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # solve for the velocity upper bound of the ring
        v_guess = 4
        v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
                          args=(len(self.initial_ids), length))[0]

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
        return Box(low=-float('inf'), high=float('inf'),
                   shape=(3, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

        # normalizers
        max_speed = 15.
        if self.env_params.additional_params['ring_length'] is not None:
            max_length = self.env_params.additional_params['ring_length'][1]
        else:
            max_length = self.k.scenario.length()

        observation = np.array([
            self.k.vehicle.get_speed(rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
             self.k.vehicle.get_speed(rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
             self.k.vehicle.get_x_by_id(rl_id)) % self.k.scenario.length()
            / max_length
        ])

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
        self.k.vehicle.set_observed(lead_id)
