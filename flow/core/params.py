class SumoParams():

    def __init__(self,
                 port=None,
                 time_step=0.1,
                 vehicle_arrangement_shuffle=False,
                 starting_position_shuffle=False,
                 emission_path="./data/",
                 rl_speed_mode='no_collide',
                 human_speed_mode='no_collide',
                 rl_lane_change_mode="no_lat_collide",
                 human_lane_change_mode="no_lat_collide",
                 sumo_binary="sumo"):
        """
        Parameters used to pass the time step and sumo-specified safety
        modes, which constrain the dynamics of vehicles in the network to
        prevent crashes. In addition, this parameter may be used to specify
        whether to use sumo's gui during the experiment's runtime

        Specify the rl_speed_mode and human_speed_mode as the SUMO-defined speed
        mode used to constrain acceleration actions.
        The available speed modes are as follows:
         - "no_collide" (default): Human and RL cars are preventing from
           reaching speeds that may cause crashes (also serves as a failsafe).
         - "aggressive": Human and RL cars are not limited by sumo with regard
           to their accelerations, and can crash longitudinally

        Specify the SUMO-defined lane-changing mode used to constrain
        lane-changing actions. The available lane-changing modes are as follows:
         - default: Human and RL cars can only safely change into lanes
         - "strategic": Human cars make lane changes in accordance with SUMO to
           provide speed boosts
         - "no_lat_collide": RL cars can lane change into any space, no matter
           how likely it is to crash
         - "aggressive": RL cars are not limited by sumo with regard to their
           lane-change actions, and can crash longitudinally

        Attributes
        ----------
        port: int, optional
            Port for Traci to connect to; finds an empty port by default
        time_step: float optional
            seconds per simulation step; 0.1 by default
        vehicle_arrangement_shuffle: bool, optional
            determines if initial conditions of vehicles are shuffled at reset;
            False by default
        starting_position_shuffle: bool, optional
            determines if starting position of vehicles should be updated
            between rollouts; False by default
        emission_path: str, optional
            Path to the folder in which to create the emissions output.
            Emissions output is not generated if this value is not specified
        rl_speed_mode: str, optional
            may be one of: 'aggressive' or 'no collide'
        human_speed_mode: str, optional
            may be one of: 'aggressive' or 'no collide'
        rl_lane_change_mode: str, optional
            may be one of 'no_lat_collide' or 'strategic' or 'aggressive'
        human_lane_change_mode: str, optional
            may be one of 'no_lat_collide' or 'strategic' or 'aggressive'
        sumo_binary: str, optional
            specifies whether to visualize the rollout(s). May be:
                - 'sumo-gui' to run the experiment with the gui
                - 'sumo' to run without the gui (default)
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
    def __init__(self,
                 longitudinal_fail_safe='None',
                 max_speed=55.0,
                 lane_change_duration=None,
                 shared_reward=False,
                 shared_policy=False,
                 additional_params=None,
                 max_deacc=-6,
                 max_acc=3):
        """
        Provides several environment and experiment-specific parameters. This
        includes specifying the parameters of the action space and relevant
        coefficients to the reward function.

        Attributes
        ----------
        longitudinal_fail_safe: str, optional
            Failsafe strategy to prevent bumper to bumper collisions; may be
            one of "None", "safe velocity", or "instantaneous"
        max_speed: float, optional
            max speed of vehicles in the simulation; defaults to 55 m/s
        lane_change_duration: float, optional
            lane changing duration is always present in the environment, but
            only used by sub-classes that apply lane changing; defaults to
            5 seconds
        shared_reward: bool, optional
            use a shared reward; defaults to False
        shared_policy: bool, optional
            use a shared policy; defaults to False
        additional_params: dict, optional
            Specify additional environment params for a specific environment
            configuration
        """
        self.fail_safe = longitudinal_fail_safe
        self.max_speed = max_speed
        self.lane_change_duration = lane_change_duration
        self.shared_reward = shared_reward
        self.shared_policy = shared_policy
        self.additional_params = additional_params
        self.max_deacc = max_deacc
        self.max_acc = max_acc


    def get_additional_param(self, key):
        return self.additional_params[key]

    def get_lane_change_duration(self, time_step):
        """
        Determines the lane change duration in units of steps

        Parameters
        ----------
        time_step: elapsed time per simulation step

        Returns
        -------
        lane_change_duration: float
            minimum number of steps in between lane changes
        """
        if not self.lane_change_duration:
            return 5 / time_step
        else:
            return self.lane_change_duration / time_step


class NetParams:
    def __init__(self,
                 net_path="debug/net/",
                 cfg_path="debug/cfg/",
                 no_internal_links=True,
                 lanes=1,
                 speed_limit=55,
                 additional_params=None):
        """
        Network configuration parameters

        Parameters
        ----------
        net_path: str, optional
            path to the network files created to create a network with sumo
        cfg_path: str, optional
            path to the config files created to create a network with sumo
        no_internal_links: bool, optional
            determines whether the space between edges is finite. Important
            when using networks with intersections; default is False
        lanes: int or dict, optional
            number of lanes for each edge in the network. May be specified as a
            single integer (in which case all lanes are assumed to have the same
            number of lanes), or a dict, ex: {"edge_1": 2, "edge_2": 1, ...}
        speed_limit: float or dict, optional
            speed limit for each edge in the network. May be specified as a
            single value (in which case all edges are assumed to have the same
            speed limit), or a dict, ex: {"edge_1": 30, "edge_2": 35, ...}
        additional_params: dict, optional
            network specific parameters; see each subclass for a description of
            what is needed
        """
        self.net_path = net_path
        self.cfg_path = cfg_path
        self.no_internal_links = no_internal_links
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.additional_params = additional_params


class InitialConfig:

    def __init__(self,
                 shuffle=False,
                 spacing="uniform",
                 scale=2.5,
                 downscale=5,
                 x0=0,
                 bunching=0,
                 lanes_distribution=1,
                 distribution_length=None,
                 positions=None,
                 lanes=None,
                 additional_params=None):
        """
        Parameters that affect the positioning of vehicle in the network at
        the start of a rollout. By default, vehicles are uniformly distributed
        in the network.

        Attributes
        ----------
        shuffle: bool, optional
            specifies whether the ordering of vehicles in the Vehicles class
            should be shuffled upon initialization.
        spacing: str, optional
            specifies the positioning of vehicles in the network relative to
            one another. May be one of:
              - "uniform" (default)
              - "gaussian"
              - "gaussian_additive"
              - "custom": a user-specified spacing method placed in
                gen_custom_spacing(). If no method is employed in the scenario,
                a NotImplementedError is returned.
        scale: float, optional
            determines the intensity of noise in the case of “gaussian” spacing
        downscale: float, optional
            determines the intensity of noise in the case of “gaussian_additive”
            spacing
        x0: float, optional
            position of the first vehicle to be placed in the network
        bunching: float, optional
            reduces the portion of the network that should be filled with
            vehicles by this amount.
        lanes_distribution: int, optional
            number of lanes vehicles should be dispersed into (cannot be greater
            than the number of lanes in the network)
        distribution_length: float, optional
            length that vehicles should be disperse in; default is network
            length
        positions: list, optional
            used if the user would like to specify user-generated initial
            positions.
        lanes: list, optional
            used if the user would like to specify user-generated initial
            positions.
        additional_params: dict, optional
            some other network-specific params
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
        if additional_params is None:
            self.additional_params = dict()
        else:
            self.additional_params = additional_params

    def get_additional_params(self, key):
        return self.additional_params[key]
