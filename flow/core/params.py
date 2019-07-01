"""Objects that define the various meta-parameters of an experiment."""

import logging
import collections

from flow.utils.flow_warnings import deprecation_warning
from flow.controllers.car_following_models import SimCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController


SPEED_MODES = {
    "aggressive": 0,
    "obey_safe_speed": 1,
    "no_collide": 7,
    "right_of_way": 25,
    "all_checks": 31
}

LC_MODES = {"aggressive": 0, "no_lat_collide": 512, "strategic": 1621}

# Traffic light defaults
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.6
SHOW_DETECTORS = True


class TrafficLightParams:
    """Base traffic light.

    This class is used to place traffic lights in the network and describe
    the state of these traffic lights. In addition, this class supports
    modifying the states of certain lights via TraCI.
    """

    def __init__(self, baseline=False):
        """Instantiate base traffic light.

        Parameters
        ----------
        baseline: bool
        """
        # traffic light xml properties
        self.__tls_properties = dict()

        # all traffic light parameters are set to default baseline values
        self.baseline = baseline

    def add(self,
            node_id,
            tls_type="static",
            programID=10,
            offset=None,
            phases=None,
            maxGap=None,
            detectorGap=None,
            showDetectors=None,
            file=None,
            freq=None):
        """Add a traffic light component to the network.

        When generating networks using xml files, using this method to add a
        traffic light will explicitly place the traffic light in the requested
        node of the generated network.

        If traffic lights are not added here but are already present in the
        network (e.g. through a prebuilt net.xml file), then the traffic light
        class will identify and add them separately.

        Parameters
        ----------
        node_id : str
            name of the node with traffic lights
        tls_type : str, optional
            type of the traffic light (see Note)
        programID : str, optional
            id of the traffic light program (see Note)
        offset : int, optional
            initial time offset of the program
        phases : list  of dict, optional
            list of phases to be followed by the traffic light, defaults
            to default sumo traffic light behavior. Each element in the list
            must consist of a dict with two keys:

            * "duration": length of the current phase cycle (in sec)
            * "state": string consist the sequence of states in the phase
            * "minDur": optional
                The minimum duration of the phase when using type actuated
            * "maxDur": optional
                The maximum duration of the phase when using type actuated

        maxGap : int, optional
            describes the maximum time gap between successive vehicle that will
            cause the current phase to be prolonged, **used for actuated
            traffic lights**
        detectorGap : int, optional
            used for actuated traffic lights
            determines the time distance between the (automatically generated)
            detector and the stop line in seconds (at each lanes maximum
            speed), **used for actuated traffic lights**
        showDetectors : bool, optional
            toggles whether or not detectors are shown in sumo-gui, **used for
            actuated traffic lights**
        file : str, optional
            which file the detector shall write results into
        freq : int, optional
            the period over which collected values shall be aggregated

        Note
        ----
        For information on defining traffic light properties, see:
        http://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
        """
        # prepare the data needed to generate xml files
        self.__tls_properties[node_id] = {"id": node_id, "type": tls_type}

        if programID:
            self.__tls_properties[node_id]["programID"] = programID

        if offset:
            self.__tls_properties[node_id]["offset"] = offset

        if phases:
            self.__tls_properties[node_id]["phases"] = phases

        if tls_type == "actuated":
            # Required parameters
            self.__tls_properties[node_id]["max-gap"] = \
                maxGap if maxGap else MAX_GAP
            self.__tls_properties[node_id]["detector-gap"] = \
                detectorGap if detectorGap else DETECTOR_GAP
            self.__tls_properties[node_id]["show-detectors"] = \
                showDetectors if showDetectors else SHOW_DETECTORS

            # Optional parameters
            if file:
                self.__tls_properties[node_id]["file"] = file

            if freq:
                self.__tls_properties[node_id]["freq"] = freq

    def get_properties(self):
        """Return traffic light properties.

        This is meant to be used by the generator to import traffic light data
        to the .net.xml file
        """
        return self.__tls_properties

    def actuated_default(self):
        """Return the default values for an actuated scenario.

        An actuated scenario is a scenario for a system where
        all junctions are actuated traffic lights.

        Returns
        -------
        tl_logic : dict
            traffic light logic
        """
        tl_type = "actuated"
        program_id = 1
        max_gap = 3.0
        detector_gap = 0.8
        show_detectors = True
        phases = [{
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "GrGr"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "yryr"
        }, {
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "rGrG"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "ryry"
        }]

        return {
            "tl_type": str(tl_type),
            "program_id": str(program_id),
            "max_gap": str(max_gap),
            "detector_gap": str(detector_gap),
            "show_detectors": show_detectors,
            "phases": phases
        }


class VehicleParams:
    """Base vehicle class.

    This is used to describe the state of all vehicles in the network.
    State information on the vehicles for a given time step can be set or
    retrieved from this class.
    """

    def __init__(self):
        """Instantiate the base vehicle class."""
        self.ids = []  # ids of all vehicles

        # vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        # Ordered dictionary used to keep neural net inputs in order
        self.__vehicles = collections.OrderedDict()

        #: total number of vehicles in the network
        self.num_vehicles = 0
        #: int : number of rl vehicles in the network
        self.num_rl_vehicles = 0
        #: int : number of unique types of vehicles in the network
        self.num_types = 0
        #: list of str : types of vehicles in the network
        self.types = []

        #: dict (str, str) : contains the parameters associated with each type
        #: of vehicle
        self.type_parameters = dict()

        #: dict (str, int) : contains the minGap attribute of each type of
        #: vehicle
        self.minGap = dict()

        #: list : initial state of the vehicles class, used for serialization
        #: purposes
        self.initial = []

    def add(self,
            veh_id,
            acceleration_controller=(SimCarFollowingController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=None,
            initial_speed=0,
            num_vehicles=1,
            car_following_params=None,
            lane_change_params=None):
        """Add a sequence of vehicles to the list of vehicles in the network.

        Parameters
        ----------
        veh_id : str
            base vehicle ID for the vehicles (will be appended by a number)
        acceleration_controller : tup, optional
            1st element: flow-specified acceleration controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        lane_change_controller : tup, optional
            1st element: flow-specified lane-changer controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        routing_controller : tup, optional
            1st element: flow-specified routing controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        initial_speed : float, optional
            initial speed of the vehicles being added (in m/s)
        num_vehicles : int, optional
            number of vehicles of this type to be added to the network
        car_following_params : flow.core.params.SumoCarFollowingParams
            Params object specifying attributes for Sumo car following model.
        lane_change_params : flow.core.params.SumoLaneChangeParams
            Params object specifying attributes for Sumo lane changing model.
        """
        if car_following_params is None:
            # FIXME: depends on simulator
            car_following_params = SumoCarFollowingParams()

        if lane_change_params is None:
            # FIXME: depends on simulator
            lane_change_params = SumoLaneChangeParams()

        type_params = {}
        type_params.update(car_following_params.controller_params)
        type_params.update(lane_change_params.controller_params)

        # If a vehicle is not sumo or RL, let the minGap be zero so that it
        # does not tamper with the dynamics of the controller
        if acceleration_controller[0] != SimCarFollowingController \
                and acceleration_controller[0] != RLController:
            type_params["minGap"] = 0.0

        # This dict will be used when trying to introduce new vehicles into
        # the network via a Flow. It is passed to the vehicle kernel object
        # during environment instantiation.
        self.type_parameters[veh_id] = \
            {"acceleration_controller": acceleration_controller,
             "lane_change_controller": lane_change_controller,
             "routing_controller": routing_controller,
             "initial_speed": initial_speed,
             "car_following_params": car_following_params,
             "lane_change_params": lane_change_params}

        # TODO: delete?
        self.initial.append({
            "veh_id":
                veh_id,
            "acceleration_controller":
                acceleration_controller,
            "lane_change_controller":
                lane_change_controller,
            "routing_controller":
                routing_controller,
            "initial_speed":
                initial_speed,
            "num_vehicles":
                num_vehicles,
            "car_following_params":
                car_following_params,
            "lane_change_params":
                lane_change_params
        })

        # This is used to return the actual headways from the vehicles class.
        # It is passed to the vehicle kernel class during environment
        # instantiation.
        self.minGap[veh_id] = type_params["minGap"]

        for i in range(num_vehicles):
            v_id = veh_id + '_%d' % i

            # add the vehicle to the list of vehicle ids
            self.ids.append(v_id)

            self.__vehicles[v_id] = dict()

            # specify the type
            self.__vehicles[v_id]["type"] = veh_id

            # update the number of vehicles
            self.num_vehicles += 1
            if acceleration_controller[0] == RLController:
                self.num_rl_vehicles += 1

        # increase the number of unique types of vehicles in the network, and
        # add the type to the list of types
        self.num_types += 1
        self.types.append({"veh_id": veh_id, "type_params": type_params})

    def get_type(self, veh_id):
        """Return the type of a specified vehicle.

        Parameters
        ----------
        veh_id : str
            vehicle ID whose type the user is querying
        """
        return self.__vehicles[veh_id]["type"]


class SimParams(object):
    """Simulation-specific parameters.

    All subsequent parameters of the same type must extend this.

    Parameters
    ----------
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    restart_instance : bool, optional
        specifies whether to restart a simulation upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    """

    def __init__(self,
                 sim_step=0.1,
                 render=False,
                 restart_instance=False,
                 emission_path=None,
                 save_render=False,
                 sight_radius=25,
                 show_radius=False,
                 pxpm=2):
        """Instantiate SimParams."""
        self.sim_step = sim_step
        self.render = render
        self.restart_instance = restart_instance
        self.emission_path = emission_path
        self.save_render = save_render
        self.sight_radius = sight_radius
        self.pxpm = pxpm
        self.show_radius = show_radius


class AimsunParams(SimParams):
    """Aimsun-specific simulation parameters.

    Extends SimParams.

    Parameters
    ----------
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    restart_instance : bool, optional
        specifies whether to restart a simulation upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    scenario_name : str, optional
        name of the scenario generated in Aimsun.
    experiment_name : str, optional
        name of the experiment generated in Aimsun
    replication_name : str, optional
        name of the replication generated in Aimsun. When loading
        an Aimsun template, this parameter must be set to the name
        of the replication to be run by the simulation; in this case,
        the scenario_name and experiment_name parameters are not
        necessary as they will be obtained from the replication name.
    centroid_config_name : str, optional
        name of the centroid configuration to load in Aimsun. This
        parameter is only used when loading an Aimsun template,
        not when generating one.
    subnetwork_name : str, optional
        name of the subnetwork to load in Aimsun. This parameter is not
        used when generating a network; it can be used when loading an
        Aimsun template containing a subnetwork in order to only load
        the objects contained in this subnetwork. If set to None or if the
        specified subnetwork does not exist, the whole network will be loaded.
    """

    def __init__(self,
                 sim_step=0.1,
                 render=False,
                 restart_instance=False,
                 emission_path=None,
                 save_render=False,
                 sight_radius=25,
                 show_radius=False,
                 pxpm=2,
                 # set to match Flow_Aimsun.ang's scenario name
                 scenario_name="Dynamic Scenario 866",
                 # set to match Flow_Aimsun.ang's experiment name
                 experiment_name="Micro SRC Experiment 867",
                 # set to match Flow_Aimsun.ang's replication name
                 replication_name="Replication 870",
                 centroid_config_name=None,
                 subnetwork_name=None):
        """Instantiate AimsunParams."""
        super(AimsunParams, self).__init__(
            sim_step, render, restart_instance, emission_path, save_render,
            sight_radius, show_radius, pxpm)
        self.scenario_name = scenario_name
        self.experiment_name = experiment_name
        self.replication_name = replication_name
        self.centroid_config_name = centroid_config_name
        self.subnetwork_name = subnetwork_name


class SumoParams(SimParams):
    """Sumo-specific simulation parameters.

    Extends SimParams.

    These parameters are used to customize a sumo simulation instance upon
    initialization. This includes passing the simulation step length,
    specifying whether to use sumo's gui during a run, and other features
    described in the Attributes below.

    Parameters
    ----------
    port : int, optional
        Port for Traci to connect to; finds an empty port by default
    sim_step : float optional
        seconds per simulation step; 0.1 by default
    emission_path : str, optional
        Path to the folder in which to create the emissions output.
        Emissions output is not generated if this value is not specified
    lateral_resolution : float, optional
        width of the divided sublanes within a lane, defaults to None (i.e.
        no sublanes). If this value is specified, the vehicle in the
        network cannot use the "LC2013" lane change model.
    no_step_log : bool, optional
        specifies whether to add sumo's step logs to the log file, and
        print them into the terminal during runtime, defaults to True
    render : str or bool, optional
        specifies whether to visualize the rollout(s)

        * False: no rendering
        * True: delegate rendering to sumo-gui for back-compatibility
        * "gray": static grayscale rendering, which is good for training
        * "dgray": dynamic grayscale rendering
        * "rgb": static RGB rendering
        * "drgb": dynamic RGB rendering, which is good for visualization

    save_render : bool, optional
        specifies whether to save rendering data to disk
    sight_radius : int, optional
        sets the radius of observation for RL vehicles (meter)
    show_radius : bool, optional
        specifies whether to render the radius of RL observation
    pxpm : int, optional
        specifies rendering resolution (pixel / meter)
    overtake_right : bool, optional
        whether vehicles are allowed to overtake on the right as well as
        the left
    seed : int, optional
        seed for sumo instance
    restart_instance : bool, optional
        specifies whether to restart a sumo instance upon reset. Restarting
        the instance helps avoid slowdowns cause by excessive inflows over
        large experiment runtimes, but also require the gui to be started
        after every reset if "render" is set to True.
    print_warnings : bool, optional
        If set to false, this will silence sumo warnings on the stdout
    teleport_time : int, optional
        If negative, vehicles don't teleport in gridlock. If positive,
        they teleport after teleport_time seconds
    num_clients : int, optional
        Number of clients that will connect to Traci
    """

    def __init__(self,
                 port=None,
                 sim_step=0.1,
                 emission_path=None,
                 lateral_resolution=None,
                 no_step_log=True,
                 render=False,
                 save_render=False,
                 sight_radius=25,
                 show_radius=False,
                 pxpm=2,
                 overtake_right=False,
                 seed=None,
                 restart_instance=False,
                 print_warnings=True,
                 teleport_time=-1,
                 num_clients=1):
        """Instantiate SumoParams."""
        super(SumoParams, self).__init__(
            sim_step, render, restart_instance, emission_path, save_render,
            sight_radius, show_radius, pxpm)
        self.port = port
        self.lateral_resolution = lateral_resolution
        self.no_step_log = no_step_log
        self.seed = seed
        self.overtake_right = overtake_right
        self.print_warnings = print_warnings
        self.teleport_time = teleport_time
        self.num_clients = num_clients


class EnvParams:
    """Environment and experiment-specific parameters.

    This includes specifying the bounds of the action space and relevant
    coefficients to the reward function, as well as specifying how the
    positions of vehicles are modified in between rollouts.

    Parameters
    ----------
    additional_params : dict, optional
        Specify additional environment params for a specific
        environment configuration
    horizon : int, optional
        number of steps per rollouts
    warmup_steps : int, optional
        number of steps performed before the initialization of training
        during a rollout. These warmup steps are not added as steps
        into training, and the actions of rl agents during these steps
        are dictated by sumo. Defaults to zero
    sims_per_step : int, optional
        number of sumo simulation steps performed in any given rollout
        step. RL agents perform the same action for the duration of
        these simulation steps.
    evaluate : bool, optional
        flag indicating that the evaluation reward should be used
        so the evaluation reward should be used rather than the
        normal reward
    clip_actions : bool, optional
        specifies whether to clip actions from the policy by their range when
        they are inputted to the reward function. Note that the actions are
        still clipped before they are provided to `apply_rl_actions`.
    """

    def __init__(self,
                 additional_params=None,
                 horizon=float('inf'),
                 warmup_steps=0,
                 sims_per_step=1,
                 evaluate=False,
                 clip_actions=True):
        """Instantiate EnvParams."""
        self.additional_params = \
            additional_params if additional_params is not None else {}
        self.horizon = horizon
        self.warmup_steps = warmup_steps
        self.sims_per_step = sims_per_step
        self.evaluate = evaluate
        self.clip_actions = clip_actions

    def get_additional_param(self, key):
        """Return a variable from additional_params."""
        return self.additional_params[key]


class NetParams:
    """Network configuration parameters.

    Unlike most other parameters, NetParams may vary drastically dependent
    on the specific network configuration. For example, for the ring road
    the network parameters will include a characteristic length, number of
    lanes, and speed limit.

    In order to determine which additional_params variable may be needed
    for a specific scenario, refer to the ADDITIONAL_NET_PARAMS variable
    located in the scenario file.

    Parameters
    ----------
    no_internal_links : bool, optional
        determines whether the space between edges is finite. Important
        when using networks with intersections; default is False
    inflows : InFlows type, optional
        specifies the inflows of specific edges and the types of vehicles
        entering the network from these edges
    osm_path : str, optional
        path to the .osm file that should be used to generate the network
        configuration files. This parameter is only needed / used if the
        OpenStreetMapScenario class is used.
    template : str, optional
        path to the .net.xml file that should be passed to SUMO. This is
        only needed / used if the NetFileScenario class is used, such as
        in the case of Bay Bridge experiments (which use a custom net.xml
        file)
    additional_params : dict, optional
        network specific parameters; see each subclass for a description of
        what is needed
    """

    def __init__(self,
                 no_internal_links=True,
                 inflows=None,
                 osm_path=None,
                 template=None,
                 additional_params=None):
        """Instantiate NetParams."""
        self.no_internal_links = no_internal_links
        if inflows is None:
            self.inflows = InFlows()
        else:
            self.inflows = inflows
        self.osm_path = osm_path
        self.template = template
        self.additional_params = additional_params or {}


class InitialConfig:
    """Initial configuration parameters.

    These parameters that affect the positioning of vehicle in the
    network at the start of a rollout. By default, vehicles are uniformly
    distributed in the network.

    Parameters
    ----------
    shuffle : bool, optional  # TODO: remove
        specifies whether the ordering of vehicles in the Vehicles class
        should be shuffled upon initialization.
    spacing : str, optional
        specifies the positioning of vehicles in the network relative to
        one another. May be one of: "uniform", "random", or "custom".
        Default is "uniform".
    min_gap : float, optional  # TODO: remove
        minimum gap between two vehicles upon initialization, in meters.
        Default is 0 m.
    x0 : float, optional  # TODO: remove
        position of the first vehicle to be placed in the network
    perturbation : float, optional
        standard deviation used to perturb vehicles from their uniform
        position, in meters. Default is 0 m.
    bunching : float, optional
        reduces the portion of the network that should be filled with
        vehicles by this amount.
    lanes_distribution : int, optional
        number of lanes vehicles should be dispersed into. If the value is
        greater than the total number of lanes on an edge, vehicles are
        spread across all lanes.
    edges_distribution : str or list of str or dict, optional
        edges vehicles may be placed on during initialization, may be one
        of:

        * "all": vehicles are distributed over all edges
        * list of edges: list of edges vehicles can be distributed over
        * dict of edges: where the key is the name of the edge to be
          utilized, and the elements are the number of cars to place on
          each edge
    additional_params : dict, optional
        some other network-specific params
    """

    def __init__(self,
                 shuffle=False,
                 spacing="uniform",
                 min_gap=0,
                 perturbation=0.0,
                 x0=0,
                 bunching=0,
                 lanes_distribution=float("inf"),
                 edges_distribution="all",
                 additional_params=None):
        """Instantiate InitialConfig.

        These parameters that affect the positioning of vehicle in the
        network at the start of a rollout. By default, vehicles are uniformly
        distributed in the network.
        """
        self.shuffle = shuffle
        self.spacing = spacing
        self.min_gap = min_gap
        self.perturbation = perturbation
        self.x0 = x0
        self.bunching = bunching
        self.lanes_distribution = lanes_distribution
        self.edges_distribution = edges_distribution
        self.additional_params = additional_params or dict()


class SumoCarFollowingParams:
    """Parameters for sumo-controlled acceleration behavior.

    Parameters
    ----------
    speed_mode : str or int, optional
        may be one of the following:

         * "right_of_way" (default): respect safe speed, right of way and
           brake hard at red lights if needed. DOES NOT respect
           max accel and decel which enables emergency stopping.
           Necessary to prevent custom models from crashing
         * "obey_safe_speed": prevents vehicles from colliding
           longitudinally, but can fail in cases where vehicles are allowed
           to lane change
         * "no_collide": Human and RL cars are preventing from reaching
           speeds that may cause crashes (also serves as a failsafe). Note:
           this may lead to collisions in complex networks
         * "aggressive": Human and RL cars are not limited by sumo with
           regard to their accelerations, and can crash longitudinally
         * "all_checks": all sumo safety checks are activated
         * int values may be used to define custom speed mode for the given
           vehicles, specified at:
           http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29

    accel : float
        see Note
    decel : float
        see Note
    sigma : float
        see Note
    tau : float
        see Note
    min_gap : float
        see minGap Note
    max_speed : float
        see maxSpeed Note
    speed_factor : float
        see speedFactor Note
    speed_dev : float
        see speedDev in Note
    impatience : float
        see Note
    car_follow_model : str
        see carFollowModel in Note
    kwargs : dict
        used to handle deprecations

    Note
    ----
    For a description of all params, see:
    http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
    """

    def __init__(
            self,
            speed_mode='right_of_way',
            accel=2.6,
            decel=4.5,
            sigma=0.5,
            tau=1.0,  # past 1 at sim_step=0.1 you no longer see waves
            min_gap=2.5,
            max_speed=30,
            speed_factor=1.0,
            speed_dev=0.1,
            impatience=0.5,
            car_follow_model="IDM",
            **kwargs):
        """Instantiate SumoCarFollowingParams."""
        # check for deprecations (minGap)
        if "minGap" in kwargs:
            deprecation_warning(self, "minGap", "min_gap")
            min_gap = kwargs["minGap"]

        # check for deprecations (maxSpeed)
        if "maxSpeed" in kwargs:
            deprecation_warning(self, "maxSpeed", "max_speed")
            max_speed = kwargs["maxSpeed"]

        # check for deprecations (speedFactor)
        if "speedFactor" in kwargs:
            deprecation_warning(self, "speedFactor", "speed_factor")
            speed_factor = kwargs["speedFactor"]

        # check for deprecations (speedDev)
        if "speedDev" in kwargs:
            deprecation_warning(self, "speedDev", "speed_dev")
            speed_dev = kwargs["speedDev"]

        # check for deprecations (carFollowModel)
        if "carFollowModel" in kwargs:
            deprecation_warning(self, "carFollowModel", "car_follow_model")
            car_follow_model = kwargs["carFollowModel"]

        # create a controller_params dict with all the specified parameters
        self.controller_params = {
            "accel": accel,
            "decel": decel,
            "sigma": sigma,
            "tau": tau,
            "minGap": min_gap,
            "maxSpeed": max_speed,
            "speedFactor": speed_factor,
            "speedDev": speed_dev,
            "impatience": impatience,
            "carFollowModel": car_follow_model,
        }

        # adjust the speed mode value
        if isinstance(speed_mode, str) and speed_mode in SPEED_MODES:
            speed_mode = SPEED_MODES[speed_mode]
        elif not (isinstance(speed_mode, int)
                  or isinstance(speed_mode, float)):
            logging.error("Setting speed mode of to default.")
            speed_mode = SPEED_MODES["obey_safe_speed"]

        self.speed_mode = speed_mode


class SumoLaneChangeParams:
    """Parameters for sumo-controlled lane change behavior.

    Parameters
    ----------
    lane_change_mode : str or int, optional
        may be one of the following:

        * "no_lat_collide" (default): Human cars will not make lane
          changes, RL cars can lane change into any space, no matter how
          likely it is to crash
        * "strategic": Human cars make lane changes in accordance with SUMO
          to provide speed boosts
        * "aggressive": RL cars are not limited by sumo with regard to
          their lane-change actions, and can crash longitudinally
        * int values may be used to define custom lane change modes for the
          given vehicles, specified at:
          http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29

    model : str, optional
        see laneChangeModel in Note
    lc_strategic : float, optional
        see lcStrategic in Note
    lc_cooperative : float, optional
        see lcCooperative in Note
    lc_speed_gain : float, optional
        see lcSpeedGain in Note
    lc_keep_right : float, optional
        see lcKeepRight in Note
    lc_look_ahead_left : float, optional
        see lcLookaheadLeft in Note
    lc_speed_gain_right : float, optional
        see lcSpeedGainRight in Note
    lc_sublane : float, optional
        see lcSublane in Note
    lc_pushy : float, optional
        see lcPushy in Note
    lc_pushy_gap : float, optional
        see lcPushyGap in Note
    lc_assertive : float, optional
        see lcAssertive in Note
    lc_impatience : float, optional
        see lcImpatience in Note
    lc_time_to_impatience : float, optional
        see lcTimeToImpatience in Note
    lc_accel_lat : float, optional
        see lcAccelLate in Note
    kwargs : dict
        used to handle deprecations

    Note
    ----
    For a description of all params, see:
    http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
    """

    def __init__(self,
                 lane_change_mode="no_lat_collide",
                 model="LC2013",
                 lc_strategic=1.0,
                 lc_cooperative=1.0,
                 lc_speed_gain=1.0,
                 lc_keep_right=1.0,
                 lc_look_ahead_left=2.0,
                 lc_speed_gain_right=1.0,
                 lc_sublane=1.0,
                 lc_pushy=0,
                 lc_pushy_gap=0.6,
                 lc_assertive=1,
                 lc_impatience=0,
                 lc_time_to_impatience=float("inf"),
                 lc_accel_lat=1.0,
                 **kwargs):
        """Instantiate SumoLaneChangeParams."""
        # check for deprecations (lcStrategic)
        if "lcStrategic" in kwargs:
            deprecation_warning(self, "lcStrategic", "lc_strategic")
            lc_strategic = kwargs["lcStrategic"]

        # check for deprecations (lcCooperative)
        if "lcCooperative" in kwargs:
            deprecation_warning(self, "lcCooperative", "lc_cooperative")
            lc_cooperative = kwargs["lcCooperative"]

        # check for deprecations (lcSpeedGain)
        if "lcSpeedGain" in kwargs:
            deprecation_warning(self, "lcSpeedGain", "lc_speed_gain")
            lc_speed_gain = kwargs["lcSpeedGain"]

        # check for deprecations (lcKeepRight)
        if "lcKeepRight" in kwargs:
            deprecation_warning(self, "lcKeepRight", "lc_keep_right")
            lc_keep_right = kwargs["lcKeepRight"]

        # check for deprecations (lcLookaheadLeft)
        if "lcLookaheadLeft" in kwargs:
            deprecation_warning(self, "lcLookaheadLeft", "lc_look_ahead_left")
            lc_look_ahead_left = kwargs["lcLookaheadLeft"]

        # check for deprecations (lcSpeedGainRight)
        if "lcSpeedGainRight" in kwargs:
            deprecation_warning(self, "lcSpeedGainRight",
                                "lc_speed_gain_right")
            lc_speed_gain_right = kwargs["lcSpeedGainRight"]

        # check for deprecations (lcSublane)
        if "lcSublane" in kwargs:
            deprecation_warning(self, "lcSublane", "lc_sublane")
            lc_sublane = kwargs["lcSublane"]

        # check for deprecations (lcPushy)
        if "lcPushy" in kwargs:
            deprecation_warning(self, "lcPushy", "lc_pushy")
            lc_pushy = kwargs["lcPushy"]

        # check for deprecations (lcPushyGap)
        if "lcPushyGap" in kwargs:
            deprecation_warning(self, "lcPushyGap", "lc_pushy_gap")
            lc_pushy_gap = kwargs["lcPushyGap"]

        # check for deprecations (lcAssertive)
        if "lcAssertive" in kwargs:
            deprecation_warning(self, "lcAssertive", "lc_assertive")
            lc_assertive = kwargs["lcAssertive"]

        # check for deprecations (lcImpatience)
        if "lcImpatience" in kwargs:
            deprecation_warning(self, "lcImpatience", "lc_impatience")
            lc_impatience = kwargs["lcImpatience"]

        # check for deprecations (lcTimeToImpatience)
        if "lcTimeToImpatience" in kwargs:
            deprecation_warning(self, "lcTimeToImpatience",
                                "lc_time_to_impatience")
            lc_time_to_impatience = kwargs["lcTimeToImpatience"]

        # check for deprecations (lcAccelLat)
        if "lcAccelLat" in kwargs:
            deprecation_warning(self, "lcAccelLat", "lc_accel_lat")
            lc_accel_lat = kwargs["lcAccelLat"]

        # check for valid model
        if model not in ["LC2013", "SL2015"]:
            logging.error("Invalid lane change model! Defaulting to LC2013")
            model = "LC2013"

        if model == "LC2013":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lc_strategic),
                "lcCooperative": str(lc_cooperative),
                "lcSpeedGain": str(lc_speed_gain),
                "lcKeepRight": str(lc_keep_right),
                # "lcLookaheadLeft": str(lcLookaheadLeft),
                # "lcSpeedGainRight": str(lcSpeedGainRight)
            }
        elif model == "SL2015":
            self.controller_params = {
                "laneChangeModel": model,
                "lcStrategic": str(lc_strategic),
                "lcCooperative": str(lc_cooperative),
                "lcSpeedGain": str(lc_speed_gain),
                "lcKeepRight": str(lc_keep_right),
                "lcLookaheadLeft": str(lc_look_ahead_left),
                "lcSpeedGainRight": str(lc_speed_gain_right),
                "lcSublane": str(lc_sublane),
                "lcPushy": str(lc_pushy),
                "lcPushyGap": str(lc_pushy_gap),
                "lcAssertive": str(lc_assertive),
                "lcImpatience": str(lc_impatience),
                "lcTimeToImpatience": str(lc_time_to_impatience),
                "lcAccelLat": str(lc_accel_lat)
            }

        # adjust the lane change mode value
        if isinstance(lane_change_mode, str) and lane_change_mode in LC_MODES:
            lane_change_mode = LC_MODES[lane_change_mode]
        elif not (isinstance(lane_change_mode, int)
                  or isinstance(lane_change_mode, float)):
            logging.error("Setting lane change mode to default.")
            lane_change_mode = LC_MODES["no_lat_collide"]

        self.lane_change_mode = lane_change_mode


class InFlows:
    """Used to add inflows to a network.

    Inflows can be specified for any edge that has a specified route or routes.
    """

    def __init__(self):
        """Instantiate Inflows."""
        self.num_flows = 0
        self.__flows = []

    def add(self,
            veh_type,
            edge,
            name="flow",
            begin=1,
            end=2e6,
            vehs_per_hour=None,
            probability=None,
            **kwargs):
        r"""Specify a new inflow for a given type of vehicles and edge.

        Parameters
        ----------
        veh_type : str
            type of vehicles entering the edge, must match one of the types set
            in the Vehicles class.
        edge : str
            starting edge for vehicles in this inflow.
        name : str, optional
            prefix for inflow vehicles
        begin : float, optional
            see Note
        end : float, optional
            see Note
        vehs_per_hour : float, optional
            see vehsPerHour in Note
        probability : float, optional
            see Note
        kwargs : dict, optional
            see Note

        Note
        ----
        For information on the parameters start, end, vehs_per_hour, period,
        probability, number, as well as other vehicle type and routing
        parameters that may be added via \*\*kwargs, refer to:
        http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
        """
        # check for deprecations (vehsPerHour)
        if "vehsPerHour" in kwargs:
            deprecation_warning(self, "vehsPerHour", "vehs_per_hour")
            vehs_per_hour = kwargs["vehsPerHour"]
            # delete since all parameters in kwargs are used again later
            del kwargs["vehsPerHour"]

        new_inflow = {
            "name": "%s_%d" % (name, self.num_flows),
            "vtype": veh_type,
            "edge": edge,
            "end": end
        }

        new_inflow.update(kwargs)

        if begin is not None:
            new_inflow["begin"] = begin
        if vehs_per_hour is not None:
            new_inflow["vehsPerHour"] = vehs_per_hour
        if probability is not None:
            new_inflow["probability"] = probability

        self.__flows.append(new_inflow)

        self.num_flows += 1

    def get(self):
        """Return the inflows of each edge."""
        return self.__flows
