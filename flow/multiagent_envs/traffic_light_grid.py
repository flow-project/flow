"""Multi-agent environments for scenarios with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.multiagent_envs.multiagent_env import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # num of nearby lights the agent can observe {0, ..., num_traffic_lights-1}
    "num_local_lights": 4,  # FIXME: not implemented yet
    # num of nearby edges the agent can observe {0, ..., num_edges}
    "num_local_edges": 4,  # FIXME: not implemented yet
}

# Index for retrieving ID when splitting node name, e.g. ":center#"
ID_IDX = 1


class MultiTrafficLightGridPOEnv(PO_TrafficLightGridEnv, MultiEnv):
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

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(3 * 4 * self.num_observed +
                   2 * self.num_local_edges +
                   3 * (1 + self.num_local_lights),
                   ),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge number.
        - For edges in the network, gives the density and average velocity.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        """
        # Normalization factors
        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # TODO(cathywu) refactor PO_TrafficLightGridEnv with convenience
        # methods for observations, but remember to flatten for single-agent

        # Observed vehicle information
        speeds = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        for _, edges in self.scenario.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
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
                local_edge_numbers.extend([self._convert_edge(
                    self.k.vehicle.get_edge(veh_id)) / (
                    self.k.scenario.network.num_edges - 1) for veh_id in
                                           observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)

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
        density = np.array(density)
        velocity_avg = np.array(velocity_avg)
        self.observed_ids = all_observed_ids

        # Traffic light information
        last_change = self.last_change.flatten()
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        # TODO(cathywu) are these values reasonable?
        last_change = np.append(last_change, [0])
        direction = np.append(direction, [0])
        currently_yellow = np.append(currently_yellow, [1])

        obs = {}
        # TODO(cathywu) allow differentiation between rl and non-rl lights
        node_to_edges = self.scenario.node_mapping
        for rl_id in self.k.traffic_light.get_ids():
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edge_numbers = [self.k.scenario.get_edge_list().index(e)
                                  for e in local_edges]
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            observation = np.array(np.concatenate(
                [speeds[rl_id_num], dist_to_intersec[rl_id_num],
                 edge_number[rl_id_num], density[local_edge_numbers],
                 velocity_avg[local_edge_numbers], last_change[local_id_nums],
                 direction[local_id_nums], currently_yellow[local_id_nums]
                 ]))
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class.

        Issues action for each traffic light agent.
        """
        for rl_id, rl_action in rl_actions.items():
            i = int(rl_id.split("center")[ID_IDX])
            if self.discrete:
                raise NotImplementedError
            else:
                # convert values less than 0.0 to zero and above to 1. 0's
                # indicate that we should not switch the direction
                action = rl_action > 0.0

            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i), state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is None:
            return {}

        if self.env_params.evaluate:
            rew = -rewards.min_delay_unscaled(self)
        else:
            rew = -rewards.min_delay_unscaled(self) \
                  + rewards.penalize_standstill(self, gain=0.2)

        # each agent receives reward normalized by number of lights
        rew /= self.num_traffic_lights

        rews = {}
        for rl_id in rl_actions.keys():
            rews[rl_id] = rew
        return rews

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)
