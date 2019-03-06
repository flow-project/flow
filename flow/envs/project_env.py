"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}


class WaveAttenuationMergePOEnv(Env):
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]
        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()
        # names of the rl vehicles controlled at any step
        self.rl_veh = []
        # used for visualization
        self.leader = []
        self.follower = []

        self.communication = []

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            # actions: acceleration, lane change, communication
            shape=(3 * self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        l = 1 + 4 * max(self.scenario.num_lanes(edge) 
                for edge in self.scenario.get_edge_list())

        return Box(low=0, high=1, shape=(l * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.communication = []
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.vehicles.get_rl_ids():
                continue
            self.apply_acceleration([rl_id], [rl_actions[i]])
            self.apply_lane_change([rl_id], [rl_actions[self.num_rl + i]])  # FIXME
            self.communication.append(rl_actions[2 * self.num_rl + i])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.leader = []
        self.follower = []
        # number of leaders and followers  = num lanes
        # length of a single vehicle's observation space
        l = 1 + 4 * max(self.scenario.num_lanes(edge) 
                        for edge in self.scenario.get_edge_list())

        leads_speed = []
        leads_head = []
        follows_speed = []
        follows_head = []

        # normalizing constants
        max_speed = self.scenario.max_speed
        max_length = self.scenario.length

        observation = [0 for _ in range(l * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.vehicles.get_speed(rl_id)
            leads_id = self.vehicles.get_lane_leaders(rl_id)
            followers = self.vehicles.get_lane_followers(rl_id)

            for lead_id in leads_id:
                if lead_id in ["", None]:
                    # in case leader is not visible
                    leads_speed.append(max_speed)
                    leads_head.append(max_length)
                else:
                    self.leader.append(lead_id)
                    leads_speed.append(self.vehicles.get_speed(lead_id))
                    leads_head.append(self.vehicles.get_headway(rl_id))

            for follower in followers:
                if follower in ["", None]:
                    # in case follower is not visible
                    follows_speed.append(0)
                    follows_head.append(max_length)
                else:
                    self.follower.append(follower)
                    follows_speed.append(self.vehicles.get_speed(follower))
                    follows_head.append(self.vehicles.get_headway(follower))

            comm = (np.sum(self.communication) - self.communication[i]) / (len(communication) -1)  # TODO: minus the ith element

            observation[l * i + 0] = this_speed / max_speed
            observation[l * i + 1: l * (i + 1) + 1 * self.num_rl] = \
                np.subtract(leads_speed, this_speed) / max_speed
            observation[l * i + 1 + 1 * self.num_rl: l * (i + 1) + 2 * self.num_rl] = \
                np.divide(leads_head, max_length)
            observation[l * i + 1 + 2 * self.num_rl: l * (i + 1) + 3 * self.num_rl] = \
                np.subtract(this_speed, follows_speed) / max_speed
            observation[l * i + 1 + 3 * self.num_rl: l * (i + 1) + 4 * self.num_rl] = \
                np.divide(follows_head, max_length)
            observation[l * i + 1 + 4 * self.num_rl] = comm

        print(observation)

        return observation

    def compute_reward(self, state, rl_actions, **kwargs):
        """See class definition.

        -1 reward for every vehicle remaining in the network"""
        return -1

    def sort_by_position(self):
        """See parent class.

        Sorting occurs by the ``get_x_by_id`` method instead of
        ``get_absolute_position``.
        """
        # vehicles are sorted by their get_x_by_id value
        sorted_ids = sorted(self.vehicles.get_ids(), key=self.get_x_by_id)
        return sorted_ids, None

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.vehicles.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
