"""Environment for training cooperative merging behaviors in a loop merge."""

from flow.envs.base_env import Env
from flow.core import rewards
from gym.spaces.box import Box
import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    # number of observable vehicles preceding the rl vehicle
    "n_preceding": 2,
    # number of observable vehicles following the rl vehicle
    "n_following": 2,
    # number of observable merging-in vehicle from the larger loop
    "n_merging_in": 2,
}


class TwoLoopsMergePOEnv(Env):
    """Environment for training cooperative merging behaviors in a loop merge.

    WARNING: only supports 1 RL vehicle

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * n_preceding: number of observable vehicles preceding the rl vehicle
    * n_following: number of observable vehicles following the rl vehicle
    * n_merging_in: number of observable merging-in vehicle from the larger
      loop

    States
        Observation space is the single RL vehicle, the 2 vehicles preceding
        it, the 2 vehicles following it, the next 2 vehicles to merge in, the
        queue length, and the average velocity of the inner and outer rings.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams. The
        actions are assigned in order of a sorting mechanism (see Sorting).

    Rewards
        Rewards system-level proximity to a desired velocity while penalizing
        variances in the headways between consecutive vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Sorting
        Vehicles in this environment are sorted by their get_x_by_id values.
        The vehicle ids are then sorted by rl vehicles, then human-driven
        vehicles.
    """

    def __init__(self, env_params, sim_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.n_preceding = env_params.additional_params["n_preceding"]
        self.n_following = env_params.additional_params["n_following"]
        self.n_merging_in = env_params.additional_params["n_merging_in"]
        self.n_obs_vehicles = \
            1 + self.n_preceding + self.n_following + self.n_merging_in

        self.obs_var_labels = \
            ["speed", "pos", "queue_length", "velocity_stats"]

        super().__init__(env_params, sim_params, scenario)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=np.inf,
            shape=(2 * self.n_obs_vehicles + 3, ),
            dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(self.vehicles.num_rl_vehicles, ),
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
        vel_reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # Use a similar weighting of of the headway reward as the velocity
        # reward
        max_cost = np.array(
            [self.env_params.additional_params["target_velocity"]
             ] * self.vehicles.num_vehicles)
        max_cost = np.linalg.norm(max_cost)
        normalization = self.scenario.length / self.vehicles.num_vehicles
        headway_reward = 0.2 * max_cost * rewards.penalize_headway_variance(
            self.vehicles, self.sorted_extra_data, normalization)
        return vel_reward + headway_reward

    def get_state(self, **kwargs):
        """See class definition."""
        vel = np.zeros(self.n_obs_vehicles)
        pos = np.zeros(self.n_obs_vehicles)

        sorted = self.sorted_extra_data
        merge_len = self.scenario.intersection_length

        # Merge stretch is pos 0.0-25.5 (ish), so actively merging vehicles
        # are sorted at the front of the list. Otherwise, vehicles closest to
        # the merge are at the end of the list (effectively reverse sorted).
        if self.get_x_by_id(sorted[0]) < merge_len and self.get_x_by_id(
                sorted[1]) < merge_len:
            if not sorted[0].startswith("merge") and \
                    not sorted[1].startswith("merge"):
                vid1 = sorted[-1]
                vid2 = sorted[-2]
            elif not sorted[0].startswith("merge"):
                vid1 = sorted[1]
                vid2 = sorted[-1]
            elif not sorted[1].startswith("merge"):
                vid1 = sorted[0]
                vid2 = sorted[-1]
            else:
                vid1 = sorted[1]
                vid2 = sorted[0]
        elif self.get_x_by_id(sorted[0]) < merge_len:
            vid1 = sorted[0]
            vid2 = sorted[-1]
        else:
            vid1 = sorted[-1]
            vid2 = sorted[-2]
        pos[-2] = self.get_x_by_id(vid1)
        pos[-1] = self.get_x_by_id(vid2)
        vel[-2] = self.vehicles.get_speed(vid1)
        vel[-1] = self.vehicles.get_speed(vid2)

        # find and eliminate all the vehicles on the outer ring
        num_inner = len(sorted)

        rl_vehID = self.vehicles.get_rl_ids()[0]
        rl_srtID, = np.where(sorted == rl_vehID)
        rl_srtID = rl_srtID[0]

        # FIXME(cathywu) hardcoded for self.num_preceding = 2
        lead_id1 = sorted[(rl_srtID + 1) % num_inner]
        lead_id2 = sorted[(rl_srtID + 2) % num_inner]
        # FIXME(cathywu) hardcoded for self.num_following = 2
        follow_id1 = sorted[(rl_srtID - 1) % num_inner]
        follow_id2 = sorted[(rl_srtID - 2) % num_inner]
        vehicles = [rl_vehID[0], lead_id1, lead_id2, follow_id1, follow_id2]

        vel[:self.n_obs_vehicles - self.n_merging_in] = np.array(
            self.vehicles.get_speed(vehicles))
        pos[:self.n_obs_vehicles - self.n_merging_in] = np.array(
            [self.get_x_by_id(veh_id) for veh_id in vehicles])

        # normalize the speed
        # FIXME(cathywu) can divide by self.max_speed
        normalized_vel = np.array(vel) / self.scenario.max_speed

        # normalize the position
        normalized_pos = np.array(pos) / self.scenario.length

        # Compute number of vehicles in the outer ring
        queue_length = np.zeros(1)
        queue_length[0] = len(sorted) - num_inner

        # Compute mean velocity on inner and outer rings
        # Note: merging vehicles count towards the inner ring stats
        vel_stats = np.zeros(2)
        vel_all = self.vehicles.get_speed(sorted)
        vel_stats[0] = np.mean(vel_all[:num_inner])
        vel_stats[1] = np.mean(vel_all[num_inner:])
        vel_stats = np.nan_to_num(vel_stats)

        return np.concatenate(
            (normalized_vel, normalized_pos, queue_length, vel_stats))

    def sort_by_position(self):
        """
        See parent class.

        Instead of being sorted by a global reference, vehicles in this
        environment are sorted with regards to which ring this currently
        reside on.
        """
        pos = [self.get_x_by_id(veh_id) for veh_id in self.vehicles.get_ids()]
        sorted_indx = np.argsort(pos)
        sorted_ids = np.array(self.vehicles.get_ids())[sorted_indx]

        sorted_human_ids = [
            veh_id for veh_id in sorted_ids
            if veh_id not in self.vehicles.get_rl_ids()
        ]

        sorted_rl_ids = [
            veh_id for veh_id in sorted_ids
            if veh_id in self.vehicles.get_rl_ids()
        ]

        sorted_separated_ids = sorted_human_ids + sorted_rl_ids

        return sorted_separated_ids, sorted_ids
