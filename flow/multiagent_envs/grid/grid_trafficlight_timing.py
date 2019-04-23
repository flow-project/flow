"""Environment for training the timinig of the traffic lights in a grid scenario."""

import numpy as np
from flow.core import rewards
from flow.multiagent_envs.multiagent_env import MultiEnv



# todo for Ashkan: This needs to be defined for multi-agent RL in grid scenario
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from flow.envs.green_wave_env import TrafficLightGridEnv

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}




class MultiAgentGrid(TrafficLightGridEnv, MultiEnv):
    """Grid multi agent env.
    """
    
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):       
        self.N_FOG_NODES = len(self.k.traffic_light.get_ids())
        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def observation_space(self):
        """
        Partially and locally observed state space.

        Velocities, distance to intersections, and traffic light state.
        """
        self.num_closest_vehicles = 3
        self.num_inbounds = 4
        speed = Box(
            low=0,
            high=1,
            shape=(num_inbounds * self.num_closest_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=1,
            shape=(num_inbounds * self.num_closest_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(num_inbounds,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, traffic_lights))

    
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

    # def get_state(self, **kwargs):
    #     state = np.array([[
    #         self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed(),
    #         self.k.vehicle.get_x_by_id(veh_id) / self.k.scenario.length()
    #     ] for veh_id in self.sorted_ids])
    #     state = np.ndarray.flatten(state)
    #     print(state, state.shape)
    #     return {'av': state, 'adversary': state}

    def get_state(self):
        """
        Returns self.num_closest_vehicles number of vehicles on each bound, closest to each traffic
        light and for each vehicle its velocity, distance to intersection. At also returns the state 
        of the 4 traffic lights in the intersection This is partially observed
        """
        speeds = []
        dist_to_intersec = []
        traffic_light_states = []
        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())
        max_dist = max(self.scenario.short_length, self.scenario.long_length,
                       self.scenario.inner_length)
        agent_state_dict = {}
        i = 0
        for intersection, edges in self.scenario.get_node_mapping():
            i = i + 1
            observed_vehicle_ids = self.k_closest_to_intersection(edges, self.num_closest_vehicles)

            speeds = [
                self.k.vehicle.get_speed(veh_id) / max_speed
                for veh_id in observed_vehicle_ids
            ]
            dist_to_intersec = [
                (self.k.scenario.edge_length(
                    self.k.vehicle.get_edge(veh_id))
                - self.k.vehicle.get_position(veh_id)) / max_dist
                for veh_id in observed_vehicle_ids
            ]

            # traffic light states
            traffic_states_chars = self.k.traffic_light.get_state(intersection)
            for j in range(self.num_inbounds):
                if traffic_states_chars[j] == 'G' or traffic_states_chars[j] == 'g': # if traffic light is green
                    traffic_light_states.append(1)
                elif traffic_states_chars[j] == 'R' or traffic_states_chars[j] == 'r': # if traffic light is red
                    traffic_light_states.append(0)
                else:
                   traffic_light_states.append(0.5) 

            # construct the state (observation) for each agent
            observation = np.array(
                np.concatenate([
                    speeds, dist_to_intersec, traffic_light_states,
                    self.last_change.flatten().tolist()
                ]))

            # each intersection is an agent, so we will make a dictionary that maps form 'intersectionI' to the state of that agent.
            agent_id = "intersection" + str(i)
            agent_state_dict.update({agent_id: observation})

        return agent_state_dict   
    
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.avg_delay_specified_vehicles(self, veh_ids)
        else:
            return rewards.desired_velocity(self, fail=kwargs["fail"])
    
    #  action_space, and additional_command are the same as 'PO_TrafficLightGridEnv'