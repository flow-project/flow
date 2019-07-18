"""Multi-agent environments for scenario with grid and AVs

These environments are used to train AVs to regulate traffic flow through an
n x m grid.
"""

from copy import deepcopy

import numpy as np
from gym.spaces.box import Box

from flow.core import rewards
from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.multiagent_envs.multiagent_env import MultiEnv

MAX_LANES = 1

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
}

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
}


class MultiGridAVsPOEnv(PO_TrafficLightGridEnv, MultiEnv):
    """Multiagent shared model version of PO_TrafficLightGridEnv.

    Required from env_params: See parent class

    States
        See parent class
    Actions
        See parent class
    Rewards
        See parent class
    Termination
        See parent class

    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = 4

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = 4

        # number of nearest edges to observe, defaults to 4
        self.traffic_lights = self.net_params.additional_params.get(
            "traffic_lights", False)

        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        self.max_speed = self.k.scenario.max_speed()

        # list of controlled edges for comparison
        outer_edges = []
        outer_edges += ["left{}_{}".format(self.rows, i) for i in range(
            self.cols)]
        # outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
        outer_edges += ["bot{}_0".format(i) for i in range(self.rows)]
        self.controlled_edges = outer_edges

    @property
    def observation_space(self):
        """State space that is partially observed.

        Local observations:
        - Observed vehicles on nearby lanes (velocity, distance to
          intersection, RL or not)
        - Local edge information (density, avg speed)
        - Ego vehicle observations (speed, max speed, headway, tailway,
          distance to intersection)
        """
        # traffic_light_obs = 3 * (1 + self.num_local_lights) * \
        #                     self.traffic_lights
        # TODO(cathywu) CHANGE
        tl_box = Box(
            low=0.,
            high=1,
            shape=(3 * self.num_local_edges * self.num_observed +
                   2 * self.num_local_edges +
                   5,
                   # traffic_light_obs,
                   ),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        add_params = self.env_params.additional_params
        max_accel = add_params.get("max_accel")
        max_decel = add_params.get("max_decel")
        # TODO(cathywu) later on, support num_lanes
        # num_lanes = self.k.scenario.num_lanes()
        return Box(
            low=-max_decel*self.sim_step, high=max_accel*self.sim_step,
            shape=(1, ), dtype=np.float32)

    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge_number, density traffic light state.
        - For edges in the network, gives the density and average velocity.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        """
        # TODO(cathywu) CHANGE
        # Normalization factors
        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # Edge information
        density = []
        velocity_avg = []
        for edge in self.k.scenario.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                # TODO(cathywu) Why is there a 5 here?
                density += [5 * len(ids) / self.k.scenario.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        density += [0]  # for the unfound edges
        velocity_avg += [1]  # for the unfound edges
        density = np.array(density)
        velocity_avg = np.array(velocity_avg)

        obs = {}
        all_observed_ids = []
        ego_edges = self.scenario.ego_edges
        for rl_id in self.k.vehicle.get_rl_ids():
            # Ego vehicle information
            ego_speed = self.k.vehicle.get_speed(rl_id) / max_speed
            ego_max_speed = self.k.vehicle.get_max_speed(rl_id) / max_speed
            ego_headway = min(self.k.vehicle.get_headway(rl_id),
                              max_dist) / max_dist
            # map no tailway (-1000) to 1.0
            ego_tailway = min(np.abs(self.k.vehicle.get_tailway(rl_id)),
                              max_dist) / max_dist
            ego_dist_to_intersec = (self.k.scenario.edge_length(
                self.k.vehicle.get_edge(rl_id)) - self.k.vehicle.get_position(
                rl_id)) / max_dist
            ego_obs = [ego_speed, ego_max_speed, ego_headway, ego_tailway,
                       ego_dist_to_intersec]

            edge = self.k.vehicle.get_edge(rl_id)
            if edge[0] == ":":  # center
                observation = np.array(np.concatenate([[0] * (
                    3 * self.num_local_edges * self.num_observed + 2 *
                    self.num_local_edges), ego_obs]))
                obs.update({rl_id: observation})
                continue

            local_edges = [edge, ego_edges[edge]['downstream_opp'],
                           ego_edges[edge]['right_incoming'],
                           ego_edges[edge]['left_incoming']]
            local_edge_numbers = []
            for local_edge in local_edges:
                try:
                    local_edge_numbers.append(
                        self.k.scenario.get_edge_list().index(local_edge))
                except ValueError:
                    # Invalid edge
                    local_edge_numbers.append(-1)

            # Observed vehicle information
            local_speeds = []
            local_dists_to_intersec = []
            local_veh_types = []
            for local_edge in local_edges:
                observed_ids = \
                    self.k_closest_to_intersection(local_edge,
                                                   self.num_observed)
                all_observed_ids.append(observed_ids)

                # check which edges we have so we can always pad in the right
                # positions
                local_speeds.extend(
                    [self.k.vehicle.get_speed(veh_id) / max_speed for veh_id in
                     observed_ids])
                local_dists_to_intersec.extend([(self.k.scenario.edge_length(
                    self.k.vehicle.get_edge(
                        veh_id)) - self.k.vehicle.get_position(
                    veh_id)) / max_dist for veh_id in observed_ids])
                local_veh_types.extend(
                    [1 if veh_id in self.k.vehicle.get_rl_ids() else 0 for
                     veh_id in observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_veh_types.extend([0] * diff)

            observation = np.array(np.concatenate(
                [local_speeds, local_dists_to_intersec, local_veh_types,
                 density[local_edge_numbers], velocity_avg[local_edge_numbers],
                 ego_obs]))

            obs.update({rl_id: observation})

        self.observed_ids = all_observed_ids

        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues new target speed for each AV agent.
        """
        for rl_id, rl_action in rl_actions.items():
            edge = self.k.vehicle.get_edge(rl_id)
            if edge:
                # If in outer lanes, on a controlled edge, in a controlled lane
                if edge[0] != ':' and edge in self.controlled_edges:
                    speed_curr = self.k.vehicle.get_speed(rl_id)
                    # keep max close to current speed
                    # FIXME(cathywu) generalize this
                    local_max = speed_curr + 5
                    max_speed_curr = self.k.vehicle.get_max_speed(rl_id)
                    next_max = np.clip(max_speed_curr + rl_action[0], 0.01,
                                       local_max)
                    self.k.vehicle.set_max_speed(rl_id, next_max)
                else:
                    # set the desired velocity of the controller to the default
                    self.k.vehicle.set_max_speed(rl_id, 23.0)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}

        rew_delay = -rewards.min_delay_unscaled(self)
        rew_still = rewards.penalize_standstill(self, gain=0.2, threshold=2.0)

        # each agent receives reward normalized by number of RL agents
        # rew /= num_agents

        rews = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if self.env_params.evaluate:
                rews[rl_id] = rew_delay
            elif rl_id in rl_actions:
                max_speed = self.k.vehicle.get_max_speed(rl_id)
                # Note: rl_action has already been applied
                curr_speed = self.k.vehicle.get_speed(rl_id)
                # control cost, also penalizes over-acceleration
                rew_speed = -0.01 * np.abs(curr_speed - max_speed)
                rews[rl_id] = rew_delay + rew_still - rew_speed
            else:
                rews[rl_id] = rew_delay + rew_still
        return rews

    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.

        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.
        """
        """See class definition."""
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        outer_edges = []
        outer_edges += ["left{}_{}".format(self.rows, i) for i in
                        range(self.cols)]
        outer_edges += ["bot{}_0".format(i) for i in range(self.rows)]
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over edges
                edge = outer_edges[self.rl_id_list.index(rl_id) % len(
                    outer_edges)]
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge=edge,
                        type_id=str('followerstopper'),
                        lane=0,
                        pos="0",
                        speed="max")
                except Exception as e:
                    print("Failed to add vehicle.", e)

        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)
