class SumoParams():
    """
    Configuration for running SUMO

    Specify the rl_speed_mode and human_speed_mode as the SUMO-defined speed
    mode used to constrain acceleration actions.
    The available speed modes are as follows:
     - "no_collide" (default): Human and RL cars are preventing from reaching speeds that may cause
                    crashes (also serves as a failsafe).
     - "aggressive": Human and RL cars are not limited by sumo with regard to their accelerations,
                     and can crash longitudinally

    Specify the SUMO-defined lane-changing mode used to constrain lane-changing actions
    The available lane-changing modes are as follows:
     - default: Human and RL cars can only safely change into lanes
     - "strategic": Human cars make lane changes in accordance with SUMO to provide speed boosts
     - "no_lat_collide": RL cars can lane change into any space, no matter how likely it is to crash
     - "aggressive": RL cars are not limited by sumo with regard to their lane-change actions,
                     and can crash longitudinally
     """

    def __init__(self, port=None, time_step=0.1, vehicle_arrangement_shuffle=False, starting_position_shuffle=False,
                 emission_path="./data/", rl_speed_mode='no_collide', human_speed_mode='no_collide',
                 rl_lane_change_mode="no_lat_collide", human_lane_change_mode="no_lat_collide",
                 sumo_binary="sumo"):
        """
        :param port: (Optional) Port for Traci to connect to; finds an empty port by default
        :param time_step: (Optional) seconds per simulation step; 0.1 by default
        :param vehicle_arrangement_shuffle: (Optional) determines if initial conditions of vehicles are shuffled
                                            at reset; False by default
        :param starting_position_shuffle: (Optional) determines if starting position of vehicles should be updated
                                          between rollouts; False by default
        :param emission_path: (Optional) Path to the folder in which to create the emissions output. Emissions output
                                          is not generated if this value is not specified
        :param rl_speed_mode:  (Optional) 'aggressive' or 'no collide'
        :param human_speed_mode: (Optional) 'aggressive' or 'no collide'
        :param rl_lane_change_mode:  (Optional) 'no_lat_collide' or 'strategic' or 'aggressive'
        :param human_lane_change_mode: (Optional) 'no_lat_collide' or 'strategic' or 'aggressive'
        :param sumo_binary: (Optional) specifies whether to visualize the rollout(s). May be:
                - 'sumo-gui' to run the experiment with the gui
                - 'sumo' to run without the gui
        """
        self.port = port
        self.time_step = time_step
        self.vehicle_arrangement_shuffle = vehicle_arrangement_shuffle
        self.starting_position_shuffle = starting_position_shuffle
        self.emission_path = emission_path
        self.rl_speed_mode = rl_speed_mode
        self.human_speed_mode = human_speed_mode
        self.rl_lane_change_mode = rl_lane_change_mode
        self.human_lane_change_mode = human_lane_change_mode
        self.sumo_binary = sumo_binary


class EnvParams:
    def __init__(self, longitudinal_fail_safe='None', observation_vel_std=0, observation_pos_std=0,
                 human_acc_std=0, rl_acc_std=0, max_speed=55.0, lane_change_duration=None,
                 shared_reward=False, shared_policy=False, additional_params=None):
        """
        :param longitudinal_fail_safe: Failsafe strategy to prevent bumper to bumper collisions
        :param observation_vel_std: observation (sensor) noise associated with velocity data
        :param observation_pos_std: observation (sensor) noise associated with position data
        :param human_acc_std: action (actuator) noise associated with human-driven vehicle acceleration
        :param rl_acc_std: action (actuator) noise associated with autonomous vehicle acceleration
        :param max_speed: max speed of vehicles in the simulation; defaults to 55 m/s
        :param lane_change_duration: lane changing duration is always present in the environment, but only used by
                                     sub-classes that apply lane changing; defaults to 5 seconds
        :param shared_reward: (Boolean) use a shared reward; defaults to False
        :param shared_policy: (Boolean) use a shared policy; defaults to False
        :param additional_params: Specify additional environment params for a specific environment configuration
        """
        self.fail_safe = longitudinal_fail_safe
        self.observation_vel_std = observation_vel_std
        self.observation_pos_std = observation_pos_std
        self.human_acc_std = human_acc_std
        self.rl_acc_std = rl_acc_std
        self.max_speed = max_speed
        self.lane_change_duration = lane_change_duration
        self.shared_reward = shared_reward
        self.shared_policy = shared_policy
        self.additional_params = additional_params

    def get_additional_param(self, key):
        return self.additional_params[key]

    def get_lane_change_duration(self, time_step):
        if not self.lane_change_duration:
            return 5 / time_step
        else:
            return self.lane_change_duration / time_step


class NetParams:
    def __init__(self, net_path="debug/net/", cfg_path="debug/cfg/", no_internal_links=True,
                 lanes=1, speed_limit=55, additional_params=None):
        """
        :param net_path: path to the network files created to create a network with sumo
        :param cfg_path: path to the configuration files created to create a network with sumo
        :param no_internal_links: determines whether the space between edges is finite. Important
                                  when using networks with intersections
        :param lanes: number of lanes for each edge in the network. May be specified as a single
                      integer (in which case all lanes are assumed to have the same number of
                      lanes), or a dict element, ex: {"edge_1": 2, "edge_2": 1, ...}
        :param speed_limit: speed limit for each edge in the network. May be specified as a single
                            value (in which case all edges are assumed to have the same speed limit),
                            or a dict element, ex: {"edge_1": 30, "edge_2": 35, ...}
        :param additional_params: network specific parameters; see each subclass for a
                                  description of what is needed
        """
        self.net_path = net_path
        self.cfg_path = cfg_path
        self.no_internal_links = no_internal_links
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.additional_params = additional_params


class InitialConfig:

    def __init__(self, shuffle=False, spacing="uniform", scale=2.5, downscale=5, x0=0, bunching=0,
                 lanes_distribution=1, distribution_length=None, positions=None, lanes=None,
                 additional_params=None):
        """

        :param shuffle: specifies whether the ordering of vehicles in the Vehicles class should be
                        shuffled upon initialization.
        :param spacing: specifies the positioning of vehicles in the network relative to one another:
                        - "uniform" (default)
                        - "gaussian"
                        - "gaussian_additive"
                        - "custom": a user-specified spacing method placed in gen_custom_spacing().
                                    If no method is employed in the scenario, a NotImplementedError
                                    is returned.
        :param scale: used in case of “gaussian” spacing
        :param downscale: used in case of “gaussian_additive” spacing
        :param x0: position of the first vehicle to be placed in the network
        :param bunching: reduces the portion of the network that should be filled with vehicles by this amount.
        :param lanes_distribution: number of lanes vehicles should be dispersed into (cannot be greater
                                   than the number of lanes in the network)
        :param distribution_length: length that vehicles should be disperse in (default is network length)
        :param positions: used if the user would like to specify user-generated initial positions
        :param lanes: used if the user would like to specify user-generated initial positions
        :param additional_params: some other network-specific params
        """
        self.shuffle = shuffle
        self.spacing = spacing
        self.scale = scale
        self.downscale = downscale
        self.x0 = x0
        self.bunching = bunching
        self.lanes_distribution = lanes_distribution
        self.distribution_length = distribution_length
        self.positions = positions
        self.lanes = lanes
        self.additional_params = additional_params

    def get_additional_params(self, key):
        return self.additional_params[key]
