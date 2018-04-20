from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # desired velocity for all vehicles in the network.
    "target_velocity": 10,
}


class AccelEnv(Env):
    """Environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:
    - target_velocity: desired velocity for all vehicles in the network.

    States
    ------
    The state consists of the velocities and absolute position of all vehicles
    in the network. This assumes a constant number of vehicles.

    Actions
    -------
    Actions are a list of acceleration for each rl vehicles, bounded by the
    maximum accelerations and decelerations specified in EnvParams.

    Rewards
    -------
    The reward function is the two-norm of the distance of the speed of the
    vehicles in the network from the "target_velocity" term. For a description
    of the reward, see: flow.core.rewards.desired_speed

    Termination
    -----------
    A rollout is terminated if the time horizon is reached or if two vehicles
    collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.
                               format(p))

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        return Box(low=-abs(self.env_params.max_decel),
                   high=self.env_params.max_accel,
                   shape=(self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,),
                  dtype=np.float32)
        return Tuple((speed, pos))

    def _apply_rl_actions(self, rl_actions):
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        scaled_pos = [self.vehicles.get_absolute_position(veh_id) /
                      self.scenario.length for veh_id in self.sorted_ids]
        scaled_vel = [self.vehicles.get_speed(veh_id) /
                      self.env_params.get_additional_param("target_velocity")
                      for veh_id in self.sorted_ids]
        state = [[vel, pos] for vel, pos in zip(scaled_vel, scaled_pos)]

        return np.array(state)

    def additional_command(self):
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)
