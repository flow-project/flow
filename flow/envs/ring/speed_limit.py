"""Environment for training the edge speed limit in a ring."""

from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum speed limit of an edge, in m/s
    'max_speed_limit': 36,
    # minimum speed limit of an edge, in m/s
    'min_speed_limit': 0,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class SpeedLimitEnv(Env):
    """Fully observed speed limit control environment.
    This environment used to train edge speed limit to improve traffic flows
    when speed limit actions are permitted by the rl agent.
    Required from env_params:
    * max_speed_limit: maximum speed limit of an edge, in m/s
    * min_speed_limit: minimum speed limit of an edge, in m/s
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment
    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.
    Actions
        Actions are a list of speed limits for each rl controlled edge, bounded
        by the maximum and minimum speed limit specified in EnvParams.
    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed
    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    Attributes
    ----------
    prev_pos : dict
        dictionary keeping track of each veh_id's previous position
    absolute_position : dict
        dictionary keeping track of each veh_id's absolute position
    obs_var_labels : list of str
        referenced in the visualizer. Tells the visualizer which
        metrics to track
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        num_rl_edges = 1
        return Box(
            low=abs(self.env_params.additional_params['min_speed_limit']),
            high=self.env_params.additional_params['max_speed_limit'],
            shape=(num_rl_edges, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        rl_edge_ids = self.k.network.get_rl_ids()
        self.k.network.set_edge_speed(rl_edge_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]

        return np.array(speed + pos)

