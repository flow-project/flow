"""Environment for training vehicles to reduce congestion in the I210."""

from copy import deepcopy
from time import time

from gym.spaces import Box, Dict, Discrete
import numpy as np

from flow.core.rewards import average_velocity
from flow.envs.multiagent.base import MultiEnv

# largest number of lanes on any given edge in the network
MAX_LANES = 6

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # whether we use an obs space that contains adjacent lane info or just the lead obs
    "lead_obs": True,
    # whether the reward should come from local vehicles instead of global rewards
    "local_reward": True
}


class I210MultiEnv(MultiEnv):
    """Partially observable multi-agent environment for the I-210 subnetworks.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicles in
        all of the preceding lanes as well, a binary value indicating which of
        these vehicles is autonomous, and the speed of the autonomous vehicle.
        Missing vehicles are padded with zeros.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right). NOTE: lane changing is currently not enabled. It's a TODO.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.lead_obs = env_params.additional_params.get("lead_obs")
        self.max_lanes = MAX_LANES

    @property
    def observation_space(self):
        """See class definition."""
        # speed, speed of leader, headway
        if self.lead_obs:
            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        # speed, dist to ego vehicle, binary value which is 1 if the vehicle is
        # an AV
        else:
            leading_obs = 3 * self.max_lanes
            follow_obs = 3 * self.max_lanes

            # speed and lane
            self_obs = 2

            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(leading_obs + follow_obs + self_obs,),
                dtype=np.float32
            )

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]

                # lane_change_softmax = np.exp(actions[1:4])
                # lane_change_softmax /= np.sum(lane_change_softmax)
                # lane_change_action = np.random.choice([-1, 0, 1],
                #                                       p=lane_change_softmax)

                self.k.vehicle.apply_acceleration(rl_id, accel)
                # self.k.vehicle.apply_lane_change(rl_id, lane_change_action)

    def get_state(self):
        """See class definition."""
        if self.lead_obs:
            veh_info = {}
            for rl_id in self.k.vehicle.get_rl_ids():
                speed = self.k.vehicle.get_speed(rl_id)
                # TODO(@evinitsky) when the vehicle is at the front of the network this might be bad.
                # Empirically I see the vehicles flip out near the end of the network
                headway = self.k.vehicle.get_headway(rl_id)
                lead_speed = self.k.vehicle.get_speed(self.k.vehicle.get_leader(rl_id))
                # If there's no lead the lead_speed is -1001.
                # I can't set this to zero, otherwise it looks like the leader is stopped!
                if lead_speed == -1001:
                    lead_speed = -100
                veh_info.update({rl_id: np.array([speed / 50.0, headway / 1000.0, lead_speed / 50.0])})
        else:
            veh_info = {rl_id: np.concatenate((self.state_util(rl_id),
                                               self.veh_statistics(rl_id)))
                        for rl_id in self.k.vehicle.get_rl_ids()}
        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        # TODO(@evinitsky) we need something way better than this. Something that adds
        # in notions of local reward
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        if self.env_params.additional_params["local_reward"]:
            des_speed = self.env_params.additional_params["target_velocity"]
            for rl_id in self.k.vehicle.get_rl_ids():
                rewards[rl_id] = 0
                speeds = []
                follow_speed = self.k.vehicle.get_speed(self.k.vehicle.get_follower(rl_id))
                if follow_speed >= 0:
                    speeds.append(follow_speed)
                if self.k.vehicle.get_speed(rl_id) >= 0:
                    speeds.append(self.k.vehicle.get_speed(rl_id))
                if len(speeds) > 0:
                    # rescale so the q function can estimate it quickly
                    rewards[rl_id] = np.mean([(des_speed - np.abs(speed - des_speed))**2 for speed in speeds]) / (des_speed**2)
        else:
            for rl_id in self.k.vehicle.get_rl_ids():
                if self.env_params.evaluate:
                    # reward is speed of vehicle if we are in evaluation mode
                    reward = self.k.vehicle.get_speed(rl_id)
                elif kwargs['fail']:
                    # reward is 0 if a collision occurred
                    reward = 0
                else:
                    # reward high system-level velocities
                    cost1 = average_velocity(self, fail=kwargs['fail'])

                    # penalize small time headways
                    cost2 = 0
                    t_min = 1  # smallest acceptable time headway

                    lead_id = self.k.vehicle.get_leader(rl_id)
                    if lead_id not in ["", None] \
                            and self.k.vehicle.get_speed(rl_id) > 0:
                        t_headway = max(
                            self.k.vehicle.get_headway(rl_id) /
                            self.k.vehicle.get_speed(rl_id), 0)
                        cost2 += min((t_headway - t_min) / t_min, 0)

                    # weights for cost1, cost2, and cost3, respectively
                    eta1, eta2 = 1.00, 0.10

                    reward = max(eta1 * cost1 + eta2 * cost2, 0)

                rewards[rl_id] = reward
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes.
        """
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)

    def state_util(self, rl_id):
        """Return an array of headway, tailway, leader speed, follower speed.

        Also return a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0 otherwise.
        If there are fewer than MAX_LANES the extra
        entries are filled with -1 to disambiguate from zeros.
        """
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl))

    def veh_statistics(self, rl_id):
        """Return speed, edge information, and x, y about the vehicle itself."""
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        return np.array([speed, lane])



class I210QMIXMultiEnv(I210MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.max_num_agents = env_params.additional_params.get("max_num_agents_qmix")
        self.num_actions = env_params.additional_params.get("num_actions")
        self.action_values = np.linspace(start=-np.abs(self.env_params.additional_params['max_decel']),
            stop=self.env_params.additional_params['max_accel'], num=self.num_actions)
        self.default_state = {idx: {"obs": -1 * np.ones(self.observation_space.spaces['obs'].shape[0]),
                               "action_mask": self.get_action_mask(valid_agent=False)}
                         for idx in range(self.max_num_agents)}
        self.rl_id_to_idx_map = {}
        self.idx_to_rl_id_map = {}

    @property
    def action_space(self):
        """See class definition."""
        return Discrete(self.num_actions + 1)

    @property
    def observation_space(self):
        obs_space = super().observation_space
        return Dict({"obs": obs_space,  "action_mask": Box(0, 1, shape=(self.action_space.n,))})

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        t = time()
        if rl_actions:
            accel_list = []
            rl_ids = []
            for idx, action in rl_actions.items():
                if idx in self.idx_to_rl_id_map.keys() and self.idx_to_rl_id_map[idx] in self.k.vehicle.get_rl_ids():
                    # 0 is the no-op
                    if action > 0:
                        accel = self.action_values[action - 1]
                        accel_list.append(accel)
                        rl_ids.append(self.idx_to_rl_id_map[idx])
            self.k.vehicle.apply_acceleration(rl_ids, accel_list)
        # print('time to apply actions is ', time() - t)

    def get_state(self, order_agents=False):
        veh_info = super().get_state()
        t = time()
        veh_info_copy = deepcopy(self.default_state)
        # print('time to make copy is ', time() - t)
        t = time()
        rl_ids = self.k.vehicle.get_rl_ids()
        if order_agents:
            abs_pos = self.k.vehicle.get_x_by_id(rl_ids)
            rl_ids = [rl_id for _, rl_id in sorted(zip(abs_pos, rl_ids))]
            rl_ids = rl_ids[::-1]
        self.rl_id_to_idx_map = {rl_id: i for i, rl_id in enumerate(rl_ids)}
        self.idx_to_rl_id_map = {i: rl_id for i, rl_id in enumerate(rl_ids)}
        veh_info_copy.update({self.rl_id_to_idx_map[rl_id]: {"obs": veh_info[rl_id],
                                          "action_mask": self.get_action_mask(valid_agent=True)}
                              for i, rl_id in enumerate(rl_ids) if i < self.max_num_agents})
        # print('time to update copy is ', time() - t)
        veh_info = veh_info_copy
        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        # There has to be one global reward for qmix
        des_speed = self.env_params.additional_params["target_velocity"]

        speeds = self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids())
        des_speed_rew = np.mean([(des_speed - np.abs(speed - des_speed)) ** 2 for speed in speeds]) / (10 * (des_speed ** 2))
        reward = np.nan_to_num(des_speed_rew)
        temp_reward_dict = {idx: reward / self.max_num_agents for idx in
                       range(self.max_num_agents)}
        # print('time to compute reward is ', time() - t)
        return temp_reward_dict

    def get_action_mask(self, valid_agent):
        """If a valid agent, return a 0 in the position of the no-op action. If not, return a 1 in that position
        and a zero everywhere else."""
        if valid_agent:
            temp_list = np.array([1 for _ in range(self.action_space.n)])
            temp_list[0] = 0
        else:
            temp_list = np.array([0 for _ in range(self.action_space.n)])
            temp_list[0] = 1
        return temp_list


class MultiStraightRoad(I210MultiEnv):
    """Partially observable multi-agent environment for a straight road. Look at superclass for more information."""

    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator)
        self.max_lanes = 1

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]

                # prevent the AV from blocking the entrance
                if self.k.vehicle.get_x_by_id(rl_id) > 100.0:
                    self.k.vehicle.apply_acceleration(rl_id, accel)


class MultiStraightRoadQMIX(I210QMIXMultiEnv):
    def get_state(self, order_agents=True):
        veh_info = super().get_state(order_agents=True)
        return veh_info

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        t = time()
        if rl_actions:
            accel_list = []
            rl_ids = []
            for idx, action in rl_actions.items():
                if idx in self.idx_to_rl_id_map.keys() and self.idx_to_rl_id_map[idx] in self.k.vehicle.get_rl_ids():
                    # 0 is the no-op
                    # prevent the AV from blocking the entrance
                    if action > 0 and self.k.vehicle.get_x_by_id(self.idx_to_rl_id_map[idx]) > 100:
                        accel = self.action_values[action - 1]
                        accel_list.append(accel)
                        rl_ids.append(self.idx_to_rl_id_map[idx])
            self.k.vehicle.apply_acceleration(rl_ids, accel_list)
        # print('time to apply actions is ', time() - t)