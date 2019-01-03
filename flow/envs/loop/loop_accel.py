"""Environment for training the acceleration behavior of vehicles in a loop."""

from flow.core import rewards
from flow.envs.base_env import Env
from flow.envs.multiagent_env import MultiEnv

from gym.spaces.box import Box

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class AccelEnv(Env):
    """Fully observed acceleration environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # initialize the list of sorted vehicle IDs
        self.sorted_ids = scenario.vehicles.get_ids()

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
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
        if self.env_params.evaluate:
            return np.mean(self.vehicles.get_speed(self.vehicles.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""
        speed = [self.vehicles.get_speed(veh_id) / self.scenario.max_speed
                 for veh_id in self.sorted_ids]
        pos = [self.get_x_by_id(veh_id) / self.scenario.length
               for veh_id in self.sorted_ids]

        return np.array(speed + pos)

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.vehicles.get_ids():
            this_pos = self.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(
                    veh_id, self.get_x_by_id(veh_id))
                self.absolute_position[veh_id] = \
                    (self.absolute_position[veh_id] + change) \
                    % self.scenario.length
                self.prev_pos[veh_id] = this_pos

        # collect list of sorted vehicle ids
        self.sorted_ids = self.sort_by_position()

    def sort_by_position(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            sorted_ids = sorted(
                self.vehicles.get_ids(),
                key=self.get_abs_position)
            return sorted_ids
        else:
            return self.vehicles.get_ids()

    def get_abs_position(self, veh_id):
        """Returns the absolute position of a vehicle."""
        return self.absolute_position[veh_id]

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        super().reset()

        for veh_id in self.vehicles.get_ids():
            self.absolute_position[veh_id] = self.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.get_x_by_id(veh_id)


class MultiAgentAccelEnv(AccelEnv, MultiEnv):
    """Adversarial multi-agent env.

    Multi-agent env with an adversarial agent perturbing
    the accelerations of the autonomous vehicle
    """
    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.vehicles.get_rl_ids()
        ]
        av_action = rl_actions['av']
        adv_action = rl_actions['adversary']
        perturb_weight = self.env_params.additional_params['perturb_weight']
        rl_action = av_action + perturb_weight * adv_action
        self.apply_acceleration(sorted_rl_ids, rl_action)

    def compute_reward(self, rl_actions, **kwargs):
        """The agents receives opposing speed rewards.

        The agent receives the class definition reward,
        the adversary receives the negative of the agent reward
        """
        if self.env_params.evaluate:
            reward = np.mean(self.vehicles.get_speed(self.vehicles.get_ids()))
            return {'av': reward, 'adversary': -reward}
        else:
            reward = rewards.desired_velocity(self, fail=kwargs['fail'])
            return {'av': reward, 'adversary': -reward}

    def get_state(self, **kwargs):
        """See class definition for the state.

        The adversary state and the agent state are identical.
        """
        # speed normalizer
        max_speed = self.scenario.max_speed
        state = np.array([[
            self.vehicles.get_speed(veh_id) / max_speed,
            self.get_x_by_id(veh_id) / self.scenario.length
        ] for veh_id in self.sorted_ids])
        state = np.ndarray.flatten(state)
        return {'av': state, 'adversary': state}
