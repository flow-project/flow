import logging


class SumoParams:
    def __init__(self,
                 port=None,
                 sim_step=0.1,
                 emission_path=None,
                 lateral_resolution=None,
                 no_step_log=True,
                 sumo_binary="sumo",
                 overtake_right=False,
                 ballistic=False,
                 seed=None,
                 cycle_time=10000):
        """
        Parameters used to pass the time step and sumo-specified safety
        modes, which constrain the dynamics of vehicles in the network to
        prevent crashes. In addition, this parameter may be used to specify
        whether to use sumo's gui during the experiment's runtime

        Attributes
        ----------
        port: int, optional
            Port for Traci to connect to; finds an empty port by default
        sim_step: float optional
            seconds per simulation step; 0.1 by default
        emission_path: str, optional
            Path to the folder in which to create the emissions output.
            Emissions output is not generated if this value is not specified
        lateral_resolution: float, optional
            width of the divided sublanes within a lane, defaults to None (i.e.
            no sublanes). If this value is specified, the vehicle in the
            network cannot use the "LC2013" lane change model.
        no_step_log: bool, optional
            specifies whether to add sumo's step logs to the log file, and
            print them into the terminal during runtime, defaults to True
        sumo_binary: str, optional
            specifies whether to visualize the rollout(s). May be:
                - 'sumo-gui' to run the experiment with the gui
                - 'sumo' to run without the gui (default)
        overtake_right: bool, optional
            whether vehicles are allowed to overtake on the right as well as
            the left
        ballistic: bool, optional
            specifies whether to use ballistic step updates. This is somewhat
            more realistic, but increases the possibility of collisions.
            Defaults to False
        seed: int, optional
            seed for sumo instance
        cycle_time: int, optional
            sets traffic lights default cycle time for base tl program
        """
        self.port = port
        self.sim_step = sim_step
        self.emission_path = emission_path
        self.lateral_resolution = lateral_resolution
        self.no_step_log = no_step_log
        self.sumo_binary = sumo_binary
        self.seed = seed
        self.ballistic = ballistic
        self.overtake_right = overtake_right
        self.cycle_time = cycle_time


class EnvParams:
    def __init__(self,
                 max_speed=55.0,  # TODO: delete me
                 lane_change_duration=None,  # TODO: move to rl vehicles only
                 vehicle_arrangement_shuffle=False,
                 starting_position_shuffle=False,
                 shared_reward=False,
                 shared_policy=False,
                 additional_params=None,
                 max_decel=-6,
                 max_accel=3,
                 horizon=500,
                 sort_vehicles=False):
        """
        Provides several environment and experiment-specific parameters. This
        includes specifying the parameters of the action space and relevant
        coefficients to the reward function.

        Attributes
        ----------
        max_speed: float, optional
            max speed of vehicles in the simulation; defaults to 55 m/s
        lane_change_duration: float, optional
            lane changing duration is always present in the environment, but
            only used by sub-classes that apply lane changing; defaults to
            5 seconds
        vehicle_arrangement_shuffle: bool, optional
            determines if initial conditions of vehicles are shuffled at reset;
            False by default
        starting_position_shuffle: bool, optional
            determines if starting position of vehicles should be updated
            between rollouts; False by default
        shared_reward: bool, optional
            use a shared reward; defaults to False
        shared_policy: bool, optional
            use a shared policy; defaults to False
        additional_params: dict, optional
            Specify additional environment params for a specific environment
            configuration
        max_decel: float, optional
            maximum deceleration of autonomous vehicles, defaults to -6 m/s2
        max_accel: float, optional
            maximum acceleration of autonomous vehicles, defaults to 3 m/s2
        horizon: int, optional
            number of steps per rollouts
        sort_vehicles: bool, optional
            specifies whether vehicles are to be sorted by position during a
            simulation step. If set to True, the environment parameter
            self.sorted_ids will return a list of all vehicles ideas sorted by
            their absolute position.
        """
        self.max_speed = max_speed
        self.lane_change_duration = lane_change_duration
        self.vehicle_arrangement_shuffle = vehicle_arrangement_shuffle
        self.starting_position_shuffle = starting_position_shuffle
        self.shared_reward = shared_reward
        self.shared_policy = shared_policy
        self.additional_params = \
            additional_params if additional_params is not None else {}
        self.max_decel = max_decel
        self.max_accel = max_accel
        self.horizon = horizon
        self.sort_vehicles = sort_vehicles

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
                 in_flows=None,
                 osm_path=None,
                 netfile=None,
                 additional_params=None):
        """
        Network configuration parameters

        Parameters
        ----------
        net_path : str, optional
            path to the network files created to create a network with sumo
        cfg_path : str, optional
            path to the config files created to create a network with sumo
        no_internal_links : bool, optional
            determines whether the space between edges is finite. Important
            when using networks with intersections; default is False
        in_flows : InFlows type, optional
            specifies the inflows of specific edges and the types of vehicles
            entering the network from these edges
        osm_path : str, optional
            path to the .osm file that should be used to generate the network
            configuration files. This parameter is only needed / used if the
            OpenStreetMapGenerator generator class is used.
        netfile : str, optional
            path to the .net.xml file that should be passed to SUMO. This is
            only needed / used if the NetFileGenerator class is used, such as
            in the case of Bay Bridge experiments (which use a custom net.xml
            file)
        additional_params : dict, optional
            network specific parameters; see each subclass for a description of
            what is needed
        """
        if additional_params is None:
            additional_params = {}
        self.net_path = net_path
        self.cfg_path = cfg_path
        self.no_internal_links = no_internal_links
        self.in_flows = in_flows
        self.osm_path = osm_path
        self.netfile = netfile
        self.additional_params = additional_params


class InitialConfig:
    def __init__(self,
                 shuffle=False,
                 spacing="uniform",
                 min_gap=0,
                 perturbation=0.0,
                 x0=0,
                 bunching=0,
                 lanes_distribution=1,
                 edges_distribution="all",
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
            one another. May be one of: "uniform", "random", or "custom".
            Default is "uniform".
        min_gap: float, optional
            minimum gap between two vehicles upon initialization, in meters.
            Default is 0 m.
        x0: float, optional
            position of the first vehicle to be placed in the network
        perturbation: float, optional
            standard deviation used to perturb vehicles from their uniform
            position, in meters. Default is 0 m.
        bunching: float, optional
            reduces the portion of the network that should be filled with
            vehicles by this amount.
        lanes_distribution: int, optional
            number of lanes vehicles should be dispersed into (cannot be greater
            than the number of lanes in the network)
        edges_distribution: list <str>, optional
            list of edges vehicles may be placed on initialization, default is
            all lanes (stated as "all")
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
        self.min_gap = min_gap
        self.perturbation = perturbation
        self.x0 = x0
        self.bunching = bunching
        self.lanes_distribution = lanes_distribution
        self.edges_distribution = edges_distribution
        self.positions = positions
        self.lanes = lanes
        if additional_params is None:
            self.additional_params = dict()
        else:
            self.additional_params = additional_params

    def get_additional_params(self, key):
        return self.additional_params[key]


class SumoCarFollowingParams:
    def __init__(self,
                 accel=2.6,
                 decel=4.5,
                 sigma=0.5,
                 tau=1.0,  # past 1 at sim_step=0.1 you no longer see waves
                 minGap=1.0,
                 maxSpeed=30,
                 speedFactor=1.0,
                 speedDev=0.0,
                 impatience=0.0,
                 carFollowModel="IDM"):
        """
        Base class for sumo-controlled acceleration behavior.

        Attributes
        ----------
        accel: float
        decel: float
        sigma: float
        tau: float
        minGap: float
        maxSpeed: float
        speedFactor: float
        speedDev: float
        impatience: float
        carFollowModel: str

        Note
        ----
        For a description of all params, see:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
        """

        # create a controller_params dict with all the specified parameters
        self.controller_params = {
            "accel": accel,
            "decel": decel,
            "sigma": sigma,
            "tau": tau,
            "minGap": minGap,
            "maxSpeed": maxSpeed,
            "speedFactor": speedFactor,
            "speedDev": speedDev,
            "impatience": impatience,
            "carFollowModel": carFollowModel,
        }


class SumoLaneChangeParams:
    def __init__(self,
                 model="LC2013",
                 lcStrategic=1.0,
                 lcCooperative=1.0,
                 lcSpeedGain=1.0,
                 lcKeepRight=1.0,
                 lcLookaheadLeft=2.0,
                 lcSpeedGainRight=1.0,
                 lcSublane=1.0,
                 lcPushy=0,
                 lcPushyGap=0.6,
                 lcAssertive=1,
                 lcImpatience=0,
                 lcTimeToImpatience=float("inf"),
                 lcAccelLat=1.0):

        if model == "LC2013":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lcStrategic),
                "lcCooperative": str(lcCooperative),
                "lcSpeedGain": str(lcSpeedGain),
                "lcKeepRight": str(lcKeepRight),
                # "lcLookaheadLeft": str(lcLookaheadLeft),
                # "lcSpeedGainRight": str(lcSpeedGainRight)
            }
        elif model == "SL2015":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lcStrategic),
                "lcCooperative": str(lcCooperative),
                "lcSpeedGain": str(lcSpeedGain),
                "lcKeepRight": str(lcKeepRight),
                "lcLookaheadLeft": str(lcLookaheadLeft),
                "lcSpeedGainRight": str(lcSpeedGainRight),
                "lcSublane": str(lcSublane),
                "lcPushy": str(lcPushy),
                "lcPushyGap": str(lcPushyGap),
                "lcAssertive": str(lcAssertive),
                "lcImpatience": str(lcImpatience),
                "lcTimeToImpatience": str(lcTimeToImpatience),
                "lcAccelLat": str(lcAccelLat)}
        else:
            logging.error("Invalid lc model! Defaulting to LC2013")
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lcStrategic),
                "lcCooperative": str(lcCooperative),
                "lcSpeedGain": str(lcSpeedGain),
                "lcKeepRight": str(lcKeepRight),
                "lcLookaheadLeft": str(lcLookaheadLeft),
                "lcSpeedGainRight": str(lcSpeedGainRight)}


class InFlows:
    def __init__(self):
        """
        Used to add inflows to a network. Inflows can be specified for any edge
        that has a specified route or routes.
        """
        self.num_flows = 0
        self.__flows = []

    def add(self, veh_type, edge, start=None, end=None, vehsPerHour=None,
            period=None, probability=None, number=None, **kwargs):
        """
        Specifies a new inflow for a given type of vehicles and edge.

        Parameters
        ----------
        veh_type: str
            type of vehicles entering the edge, must match one of the types set
            in the Vehicles class.
        edge: str
            starting edge for vehicles in this inflow.
        start: float, optional
            see Note
        end: float, optional
            see Note
        vehsPerHour: float, optional
            see Note
        period: float, optional
            see Note
        probability: float, optional
            see Note
        number: int, optional
            see Note
        kwargs: dict, optional
            see Note

        Note
        ----
        For information on the parameters start, end, vehsPerHour, period,
        probability, number, as well as other vehicle type and routing
        parameters that may be added via **kwargs, refer to:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
        """
        new_inflow = {"name": "flow_%d" % self.num_flows, "vtype": veh_type,
                      "route": "route" + edge}

        new_inflow.update(kwargs)

        if start is not None:
            new_inflow["start"] = start
        if end is not None:
            new_inflow["end"] = end
        if vehsPerHour is not None:
            new_inflow["vehsPerHour"] = vehsPerHour
        if period is not None:
            new_inflow["period"] = period
        if probability is not None:
            new_inflow["probability"] = probability
        if number is not None:
            new_inflow["number"] = number

        self.__flows.append(new_inflow)

        self.num_flows += 1

    def get(self):
        return self.__flows
