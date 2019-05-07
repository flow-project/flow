"""Environment for training the timinig of the traffic lights in a grid scenario."""

import numpy as np
from flow.core import rewards
from flow.multiagent_envs.multiagent_env import MultiEnv




# todo for Ashkan: This needs to be defined for multi-agent RL in grid scenario
from gym.spaces.discrete import Discrete
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
        
        super().__init__(env_params, sim_params, scenario, simulator)
        self.NUM_FOG_NODES = len(self.k.traffic_light.get_ids())
        self.agent_name_prefix = 'intersection'


    @property
    def action_space(self):
        """See class definition."""
        if self.discrete: 
            # each intersection is an agent, and the action is simply 0 or 1. 0 means left-right traffic passes
            # and, 1 means top-bottom traffic passes
            return Discrete(2)  
        else:
            return Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32)

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
            shape=(self.num_inbounds * self.num_closest_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0,
            high=1,
            shape=(self.num_inbounds * self.num_closest_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0,
            high=3,
            shape=(1,), # the state of this intersection. 4 possibilities. Either left-right, or top-bottom traffic is passing, or it's the yellow light between them
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, traffic_lights))

    def get_state(self):
        """
        Returns self.num_closest_vehicles number of vehicles on each bound, closest to each traffic
        light and for each vehicle its velocity, distance to intersection. At also returns the state 
        of the 4 traffic lights in the intersection This is partially observed
        """

        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())

        max_dist = max(self.scenario.short_length, self.scenario.long_length,
                    self.scenario.inner_length)
        agent_state_dict = {}
        i = 0
        for intersection, edges in self.scenario.get_node_mapping():
            i = i + 1
            agent_id = self.agent_name_prefix + str(i) 
            observed_vehicle_ids = self.k_closest_to_intersection(edges, self.num_closest_vehicles)

            speeds = []
            dist_to_intersec = []
            traffic_light_states = []

            speeds = [
                self.k.vehicle.get_speed(veh_id) / max_speed
                for veh_id in observed_vehicle_ids
            ]

            for veh_id in observed_vehicle_ids:
                if veh_id == 0:
                    dist_to_intersec.append(-1)
                else:
                    dist_to_intersec.append(
                        (self.k.scenario.edge_length(
                            self.k.vehicle.get_edge(veh_id))
                            - self.k.vehicle.get_position(veh_id)) / max_dist
                    )

            traffic_light_states_chars = self.k.traffic_light.get_state(intersection) 
            
            
            if traffic_light_states_chars == 'GrGr' :
                traffic_light_states.append(1)
            elif traffic_light_states_chars == 'rGrG': 
                traffic_light_states.append(0)
            elif traffic_light_states_chars == 'yryr' :
                traffic_light_states.append(2) 
            else: # ryry
                traffic_light_states.append(3) 

            
            # construct the state (observation) for each agent
            observation = np.array(
                np.concatenate([
                    speeds, dist_to_intersec, traffic_light_states  # or:   speeds, dist_to_intersec, self.last_change.flatten().tolist()
                ]))
            # observation = np.ndarray.flatten(observation)

            # each intersection is an agent, so we will make a dictionary that maps form "self.agent_name_prefix+'i'" to the state of that agent.
            agent_state_dict.update({agent_id: observation})

        return agent_state_dict   

    def _apply_rl_actions(self, rl_actions):

        for agent_name in rl_actions:
            action = rl_actions[agent_name]
            # check if the action space is discrete
            
            if self.discrete:
                # convert single value (Discrete) to list of 0's and 1's
                action = [int(x) for x in list('{0:0b}'.format(action))]
            else:
                # convert values less than 0.5 to zero and above to 1. 0's indicate
                # that should not switch the direction
                action = action > 0.5
            agent_num = agent_name[len(self.agent_name_prefix) : len(agent_name)] # agent_id = '15' if agent_name is 'intersection15'
            tl_num = int(agent_num, 10) - 1 # the index of traffic lights starts from 0
           
            # check if our timer has exceeded the yellow phase, meaning it
            # should switch to red
            if self.last_change[tl_num, 2] == 0:  # currently yellow
                self.last_change[tl_num, 0] += self.sim_step
                if self.last_change[tl_num, 0] >= self.min_switch_time:
                    if self.last_change[tl_num, 1] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center' + str(tl_num),
                            state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center' + str(tl_num),
                            state='rGrG')
                    self.last_change[tl_num, 2] = 1
            else:
                if action:
                    if self.last_change[tl_num, 1] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center' + str(tl_num),
                            state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center' + str(tl_num),
                            state='ryry')
                    self.last_change[tl_num, 0] = 0.0
                    self.last_change[tl_num, 1] = not self.last_change[tl_num, 1]
                    self.last_change[tl_num, 2] = 0

    def compute_reward(self, rl_actions, **kwargs):
        """Each agents receives a reward that is with regards
        to the delay of the vehicles it observers
        """
        # in the warmup steps
        if rl_actions is None:
            return {}
        i = 0
        agent_reward_dict = {}
        for intersection, edges in self.scenario.get_node_mapping():
            i = i + 1
            agent_id = self.agent_name_prefix + str(i)

            observed_vehicle_ids = self.k_closest_to_intersection(edges, self.num_closest_vehicles)
            # construct the reward for each agent
            reward = np.mean(self.k.vehicle.get_speed(observed_vehicle_ids))    # or:      reward = - rewards.avg_delay_specified_vehicles(self, observed_vehicle_ids)
            # each intersection is an agent, so we will make a dictionary that maps form "self.agent_name_prefix+'i'" to the reward of that agent.
            agent_reward_dict.update({agent_id: reward})

        if self.env_params.evaluate:
            return agent_reward_dict
        else:
            reward = rewards.desired_velocity(self, fail=kwargs['fail'])
            return agent_reward_dict
