"""Environment for training vehicles to reduce congestion in a merge."""

from copy import deepcopy

from flow.envs.multiagent.base import MultiEnv
from flow.core import rewards
from gym.spaces.box import Box
import numpy as np


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
}


class MultiAgentMergePOEnv(MultiEnv):
    """Partially observable multi-agent merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

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

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-5, high=5, shape=(5,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for rl_id in enumerate(self.k.vehicle.get_rl_ids()):
            if rl_id not in rl_actions.keys():
                # the vehicle just entered, so ignore
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        observation = {}
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation[rl_id] = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
                (this_speed - follow_speed) / max_speed,
                follow_head / max_length
            ])

        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.k.vehicle.get_rl_ids():
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1 and cost2, respectively
            eta1, eta2 = 1.00, 0.10

            reward = max(eta1 * cost1 + eta2 * cost2, 0)
            return {key: reward for key in self.k.vehicle.get_rl_ids()}

    def additional_command(self):
        """See parent class.

        This method defines which vehicles are observed for visualization
        purposes.
        """
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self, new_inflow_rate=None):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()


class MultiAgentZSCMergePOEnv(MultiEnv):
    """Partially observable multi-agent merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * return_all_vehicle_states: if true we return states for both the human driven and the
                                 RL cars

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding the vehicle, as
        well as the ego speed of the autonomous vehicles. Additionally, we find the
        vehicles on the opposite edge and return the up-to-two vehicles that are closer to the merge
        than the ego vehicle. i.e. if we are on edge 1 and 200 meters away from the merge and there
        are four vehicles on edge 2 that are 220, 180, 140, 100 meters away from the merge we will return
        the distance to the merge and speed of the vehicles that are 180 and 140 meters away.
        However, if there are no vehicles closer to the merge on the opposite edge, we will return the
        vehicle closest to the merge i.e. if the vehicle on edge 1 is 200 meters away from the merge
        and the vehicles on edge 2 are at 300, 260, and 240 meters from the merge we will return the state
        of vehicle 240 meters from the merge and pad with zeros for the missing second vehicle.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []
        self.veh_id_list = deepcopy(self.k.vehicle.get_ids())


    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-10, high=10, shape=(7,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for rl_id in enumerate(self.k.vehicle.get_rl_ids()):
            if rl_id not in rl_actions.keys():
                # the vehicle just entered, so ignore
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        observation = {}
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        # vehicles on the bottom edge
        bottom_ids = self.k.vehicle.get_ids_by_edge(['bottom'])
        bottom_positions = [(self.k.vehicle.get_position(id), id) for id in bottom_ids]
        bottom_positions = sorted(bottom_positions, key=lambda x: x[0])
        bottom_ids = [val[1] for val in bottom_positions]
        bottom_positions = [val[0] for val in bottom_positions]

        left_ids = self.k.vehicle.get_ids_by_edge(['left'])
        left_positions = [(self.k.vehicle.get_position(id), id) for id in left_ids]
        left_positions = sorted(left_positions, key=lambda x: x[0])
        left_ids = [val[1] for val in left_positions]
        left_positions = [val[0] for val in left_positions]

        if self.env_params.additional_params["return_all_vehicle_states"]:
            ids_to_return = self.k.vehicle.get_ids()
        else:
            ids_to_return = self.k.vehicle.get_rl_ids()

        for veh_id in ids_to_return:
            this_speed = self.k.vehicle.get_speed(veh_id)
            lead_id = self.k.vehicle.get_leader(veh_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(veh_id) \
                    - self.k.vehicle.get_length(veh_id)

            observation[veh_id] = np.array([
                this_speed / max_speed,
                (lead_speed - this_speed) / max_speed,
                lead_head / max_length,
            ])
            # TODO(ev) make this faster and cleaner, this is MVP code
            # okay now get the vehicle on the opposite edge
            concat_object = np.zeros(4)
            if self.k.vehicle.get_edge(veh_id) == 'left':
                veh_pos = self.k.vehicle.get_position(veh_id)
                # find the position in the list
                pos_index = np.searchsorted(bottom_positions, veh_pos)
                # we are closer than any of the vehicles on that edge
                if len(bottom_positions) > 0 and pos_index == len(bottom_positions):
                    concat_object[0:2] = [self.k.vehicle.get_position(bottom_ids[-1]) / max_length,
                                          self.k.vehicle.get_speed(bottom_ids[-1]) / max_speed]
                # there are vehicles closer than us so return their state
                for i, bottom_id in enumerate(bottom_ids[pos_index: pos_index + 2]):
                    concat_object[2 * i: 2 * (i + 1)] = [self.k.vehicle.get_position(bottom_id) / max_length,
                                                         self.k.vehicle.get_speed(bottom_id) / max_speed]
            # TODO(ev) remove copypasta
            elif self.k.vehicle.get_edge(veh_id) == 'bottom':
                veh_pos = self.k.vehicle.get_position(veh_id)
                # find the position in the list
                pos_index = np.searchsorted(left_positions, veh_pos)
                # we are closer than any of the vehicles on that edge
                if len(left_positions) > 0 and pos_index == len(left_positions):
                    concat_object[0:2] = [self.k.vehicle.get_position(left_ids[-1]),
                                          self.k.vehicle.get_speed(left_ids[-1])]
                # there are vehicles closer than us so return their state
                for i, left_id in enumerate(left_ids[pos_index: pos_index + 2]):
                    concat_object[2 * i: 2 * (i + 1)] = [self.k.vehicle.get_position(left_id) / max_length,
                                                       self.k.vehicle.get_speed(left_id) / max_speed]
            observation[veh_id] = np.concatenate((observation[veh_id], concat_object))


        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            if self.env_params.additional_params["return_all_vehicle_states"]:
                ids_to_return = self.k.vehicle.get_ids()
            else:
                ids_to_return = self.k.vehicle.get_rl_ids()
            for rl_id in ids_to_return:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1 and cost2, respectively
            eta1, eta2 = 1.00, 0.10

            reward = max(eta1 * cost1 + eta2 * cost2, 0)
            return {key: reward for key in self.k.vehicle.get_rl_ids()}

    def additional_command(self):
        """See parent class.

        This method defines which vehicles are observed for visualization
        purposes.
        """
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

        # put vehicles back if they've exited
        # if the number of rl vehicles has decreased introduce it back in
        num_veh = self.k.vehicle.num_vehicles
        if num_veh != len(self.veh_id_list):
            # find the vehicles that have exited
            diff_list = list(
                set(self.veh_id_list).difference(self.k.vehicle.get_ids()))
            for veh_id in diff_list:
                # distribute rl cars evenly over lanes
                # reintroduce it at the start of the network
                try:
                    if np.random.uniform() < 0.5:
                        insert_edge = 'inflow_highway'
                    else:
                        insert_edge = 'inflow_merge'
                    if veh_id in self.k.vehicle.get_rl_ids():
                        type_id = 'rl'
                    else:
                        type_id = 'human'
                    self.k.vehicle.add(
                        veh_id=veh_id,
                        edge=insert_edge,
                        type_id=str(type_id),
                        lane=str(0),
                        pos="0",
                        speed="max")
                except Exception:
                    pass

    def reset(self, new_inflow_rate=None):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []

        obs = super().reset()
        # id list used to keep track of which vehicles should be in the system
        self.veh_id_list = deepcopy(self.k.vehicle.get_ids())
        return obs