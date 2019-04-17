"""Environment for training the timinig of the traffic lights in a grid scenario."""

import numpy as np
from flow.core import rewards
from flow.multiagent_envs.multiagent_env import MultiEnv



# todo for Ashkan: This needs to be defined for multi-agent RL in grid scenario
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from flow.envs.green_wave_env import PO_TrafficLightGridEnv

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}

class MultiAgentGrid(PO_TrafficLightGridEnv, MultiEnv):
    """Grid multi agent env.
    """
    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        av_action = rl_actions['av']
        adv_action = rl_actions['adversary']
        perturb_weight = self.env_params.additional_params['perturb_weight']
        rl_action = av_action + perturb_weight * adv_action
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_action)

    def compute_reward(self, rl_actions, **kwargs):
        """The agents receives opposing speed rewards.

        The agent receives the class definition reward,
        the adversary receives the negative of the agent reward
        """
        if self.env_params.evaluate:
            reward = np.mean(self.k.vehicle.get_speed(
                self.k.vehicle.get_ids()))
            return {'av': reward, 'adversary': -reward}
        else:
            reward = rewards.desired_velocity(self, fail=kwargs['fail'])
            return {'av': reward, 'adversary': -reward}

    def get_state(self, **kwargs):
        state = np.array([[
            self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed(),
            self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
        ] for veh_id in self.sorted_ids])
        state = np.ndarray.flatten(state)
        print(state, state.shape)
        return {'av': state, 'adversary': state}

    
    """Environment used to train traffic lights to regulate traffic flow
    through an n x m grid.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * num_observed: number of vehicles nearest each intersection that is
      observed in the state space; defaults to 2

    States
        An observation is the number of observe vehicles in each intersection
        closest to the traffic lights, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the delay of each vehicle minus a penalty for switching
        traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.

    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []

    @property
    def observation_space(self):
        """
        Partially and locally observed state space.

        Velocities, distance to intersections, edge number (for nearby
        vehicles), and traffic light state.
        """
        num_closest_vehicles = 3
        num_inbounds = 4
        speed = Box(
            low=0,
            high=1,
            shape=(num_inbounds * num_closest_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=1,
            shape=(num_inbounds * num_closest_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(num_inbounds,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, traffic_lights))

    def get_state(self):
        """
        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        edge_number traffic light state. This is partially observed
        """
        speeds = []
        dist_to_intersec = []
        edge_number = []
        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())
        max_dist = max(self.scenario.short_length, self.scenario.long_length,
                       self.scenario.inner_length)
        all_observed_ids = []

        for node, edges in self.scenario.get_node_mapping():
            for edge in edges:
                observed_ids = \
                    self.k_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.scenario.edge_length(
                        self.k.vehicle.get_edge(veh_id))
                     - self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                edge_number += \
                    [self._convert_edge(self.k.vehicle.get_edge(veh_id))
                     / (self.k.scenario.network.num_edges - 1)
                     for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    edge_number += [0] * diff

        # now add in the density and average velocity on the edges
        density = []
        velocity_avg = []
        for edge in self.k.scenario.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                density += [5 * len(ids) / self.k.scenario.edge_length(edge)]
                velocity_avg += [
                    np.mean(
                        [self.k.vehicle.get_speed(veh_id)
                         for veh_id in ids]) / max_speed
                ]
            else:
                density += [0]
                velocity_avg += [0]
        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                speeds, dist_to_intersec, edge_number, density, velocity_avg,
                self.last_change.flatten().tolist()
            ]))
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.avg_delay_specified_vehicles(self, veh_ids)
        else:
            return rewards.desired_velocity(self, fail=kwargs["fail"])
    
    #  action_space, and additional_command are the same as 'PO_TrafficLightGridEnv'