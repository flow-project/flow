from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np


class TwoLoopsMergeEnv(Env):
    """Environment for training cooperative merging behavior in a closed loop
    merge setting.

    States
    ------
    The state consists of the velocities, and x_by_id positions of all vehicles
    in the network. This assumes a constant number of vehicles.

    Actions
    -------
    Actions are a list of acceleration for each rl vehicles, bounded by the
    maximum accelerations and decelerations specified in EnvParams. The actions
    are assigned in order of a sorting mechanism (see Sorting).

    Reward
    ------
    The reward function is the two-norm of the distance of the speed of the
    vehicles in the network from a desired speed.

    Termination
    -----------
    A rollout is terminated if the time horizon is reached or if two vehicles
    collide into one another.

    Sorting
    -------
    Vehicles in this environment are sorted by their get_x_by_id values. The
    vehicle ids are then sorted by rl vehicles, then human-driven vehicles.
    """
    @property
    def action_space(self):
        return Box(low=-np.abs(self.env_params.max_decel),
                   high=self.env_params.max_accel,
                   shape=(self.vehicles.num_rl_vehicles,))

    @property
    def observation_space(self):
        self.obs_var_labels = ["speed", "pos"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, pos))

    def apply_rl_actions(self, rl_actions):
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        return rewards.desired_velocity(self, fail=kwargs["fail"])

    def get_state(self, **kwargs):
        vel = self.vehicles.get_speed(self.sorted_ids)
        pos = [self.get_x_by_id(veh_id) for veh_id in self.sorted_ids]
        return np.array([vel, pos]).T

    def sort_by_position(self):
        """ Vehicles in this environment are sorted with regards to which ring
        this currently reside on, and then by whether they are human-driven or
        rl vehicles."""
        pos = [self.get_x_by_id(veh_id) for veh_id in self.vehicles.get_ids()]
        sorted_indx = np.argsort(pos)
        sorted_ids = np.array(self.vehicles.get_ids())[sorted_indx]

        sorted_human_ids = [veh_id for veh_id in sorted_ids
                            if veh_id not in self.vehicles.get_rl_ids()]

        sorted_rl_ids = [veh_id for veh_id in sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        sorted_separated_ids = sorted_human_ids + sorted_rl_ids

        return sorted_separated_ids, sorted_ids


class TwoLoopsMergePOEnv(TwoLoopsMergeEnv):
    """POMDP version of two-loop merge env

    """
    @property
    def observation_space(self):
        """
        See parent class.

        Observation space is the single RL vehicle, the 2 vehicles preceding it,
        the 2 vehicles following it, the next 2 vehicles to merge in, the queue
        length, and the average velocity of the inner and outer rings.

        WARNING: only supports 1 RL vehicle
        """
        self.n_preceding = 2  # FIXME(cathywu) see below
        self.n_following = 2  # FIXME(cathywu) see below
        self.n_merging_in = 2
        self.n_obs_vehicles = 1 + self.n_preceding + self.n_following + \
                              self.n_merging_in
        self.obs_var_labels = ["speed", "pos", "queue_length", "velocity_stats"]
        speed = Box(low=0, high=np.inf, shape=(self.n_obs_vehicles,))
        absolute_pos = Box(low=0., high=np.inf, shape=(self.n_obs_vehicles,))
        # dist_to_merge = Box(low=-1, high=1, shape=(1,))
        queue_length = Box(low=0, high=np.inf, shape=(1,))
        vel_stats = Box(low=-np.inf, high=np.inf, shape=(2,))
        return Tuple((speed, absolute_pos, queue_length, vel_stats))

    @property
    def action_space(self):
        if (self.scenario.net_params.additional_params.get("outer_lanes", 1) > 1 and
                    self.scenario.net_params.additional_params.get("inner_lanes", 1) > 1):
            """
            Actions are a set of accelerations from max-deacc to max-acc for each
            rl vehicle and lane changes.
            """
            return self.lane_change_action_space()
        else:
            """
            Actions are a set of accelerations from max-deacc to max-acc for each
            rl vehicle.
            """
            return Box(low=-np.abs(self.env_params.max_decel),
                       high=self.env_params.max_accel,
                       shape=(self.vehicles.num_rl_vehicles,))

    def lane_change_action_space(self):
        """
        See parent class

        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) lane-change action from -1 to 1, used to determine the
           lateral direction the vehicle will take.
        """
        max_decel = self.env_params.max_decel
        max_accel = self.env_params.max_accel

        lb = [-abs(max_decel), -1] * self.vehicles.num_rl_vehicles
        ub = [max_accel, 1] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands for the duration of the lane
        change and return negative rewards for actions during that lane change.
        if a lane change isn't applied, and sufficient time has passed, issue an
        acceleration like normal.
        """
        if (self.scenario.net_params.additional_params.get("outer_lanes", 1) > 1 and
                    self.scenario.net_params.additional_params.get("inner_lanes", 1) > 1):
            acceleration = rl_actions[::2]
            direction = np.round(rl_actions[1::2]).clip(min=-1, max=1)

            # re-arrange actions according to mapping in observation space
            sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                             if veh_id in self.vehicles.get_rl_ids()]

            # represents vehicles that are allowed to change lanes
            non_lane_changing_veh = \
                [self.time_counter <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
                 for veh_id in sorted_rl_ids]
            # vehicle that are not allowed to change have their directions set to 0
            direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

            self.apply_acceleration(sorted_rl_ids, acc=acceleration)
            self.apply_lane_change(sorted_rl_ids, direction=direction)
        else:
            sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                             if veh_id in self.vehicles.get_rl_ids()]
            self.apply_acceleration(sorted_rl_ids, rl_actions)

    def get_state(self, **kwargs):
        """
        See parent class and defined observation_space.
        """
        vel = np.zeros(self.n_obs_vehicles)
        pos = np.zeros(self.n_obs_vehicles)

        sorted = self.sorted_extra_data
        merge_len = self.scenario.intersection_length

        # FIXME(cathywu) hardcoded for observing 2 merging vehicles
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
                # print("actively merging", vid1, vid2)
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
        for i in range(len(sorted) - 1, -1, -1):
            if not sorted[i].startswith("merge"):
                num_inner = i + 1
                break

        rl_vehID = self.vehicles.get_rl_ids()[0]
        rl_srtID, = np.where(sorted == rl_vehID)
        rl_srtID = rl_srtID[0]

        # FIXME(cathywu) hardcoded for self.num_preceding = 2
        lead_id1 = sorted[(rl_srtID + 1) % num_inner]
        lead_id2 = sorted[(rl_srtID + 2) % num_inner]
        # FIXME(cathywu) hardcoded for self.num_following = 2
        follow_id1 = sorted[(rl_srtID - 1) % num_inner]
        follow_id2 = sorted[(rl_srtID - 2) % num_inner]
        vehicles = [rl_vehID, lead_id1, lead_id2, follow_id1, follow_id2]

        vel[:self.n_obs_vehicles - self.n_merging_in] = np.array(
            self.vehicles.get_speed(vehicles))
        pos[:self.n_obs_vehicles - self.n_merging_in] = np.array(
            [self.get_x_by_id(veh_id) for veh_id in vehicles])

        # normalize the speed
        # FIXME(cathywu) can divide by self.max_speed
        normalized_vel = np.array(vel) / 30.

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

        # Useful debug statements for analyzing experiment results
        # print("XXX obs", vel, pos, queue_length, vel_stats)
        # print("XXX nobs", normalized_vel, normalized_pos, queue_length,
        #       vel_stats)

        # print("XXX mean vel", np.mean(vel_all))
        # pos_all = [self.get_x_by_id(id) for id in sorted]
        # print("XXX pos", pos_all)
        # print("XXX vel", vel_all)
        # headway_all = [self.vehicles.get_headway(id) for id in sorted]
        # print("XXX head", headway_all)
        # print("XXX", sorted)

        return np.array([normalized_vel, normalized_pos, queue_length,
                         vel_stats]).T

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        Rewards high system-level velocities and large headways.
        """
        vel_reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # Use a similar weighting of of the headway reward as the velocity
        # reward
        max_cost = np.array([self.env_params.additional_params[
                                 "target_velocity"]] *
                            self.vehicles.num_vehicles)
        max_cost = np.linalg.norm(max_cost)
        normalization = self.scenario.length / self.vehicles.num_vehicles
        headway_reward = 0.2 * max_cost * rewards.penalize_headway_variance(
            self.vehicles, self.sorted_extra_data, normalization)
        # print("Rewards", vel_reward, headway_reward)
        return vel_reward + headway_reward


class TwoLoopsMergeNoRLPOEnv(TwoLoopsMergePOEnv):
    """
    POMDP Merge env compatible for SUMO-only run
    """

    def get_state(self, **kwargs):
        return np.zeros(1)
