"""Script containing the base vehicle kernel class."""
from flow.core.kernel.vehicle.base import KernelVehicle
import collections
import numpy as np
from copy import deepcopy
from flow.utils.aimsun.struct import InfVeh
from flow.controllers.car_following_models import SimCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController

# colors for vehicles
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)


class AimsunKernelVehicle(KernelVehicle):
    """Flow vehicle kernel.

    This kernel sub-class is used to interact with the simulator with regards
    to all vehicle-dependent components. Specifically, this class contains
    methods for:

    * Interacting with the simulator: This includes apply acceleration, lane
      change, and routing commands to specific vehicles in the simulator. In
      addition, methods exist to add or remove a specific vehicle from the
      network, and update internal state information after every simulation
      step in order to support and potentially speed up all state-acquisition
      methods.
    * Visually distinguishing vehicles by type: In the case when some vehicles
      are controlled by a reinforcement learning agent or some other
      controller, these methods can be used to visually distinguish the
      vehicles during rendering by RL/actuated, human-observed, and
      human-unobserved. The traci simulator, for instance, renders RL vehicles
      as red, observed human vehicles as cyan, and unobserved human vehicles as
      white. In the absence of RL/actuated agents, all vehicles are white.
    * State acquisition: Finally, this methods contains several methods for
      acquiring state information from specific vehicles. For example, if you
      would like to get the speed of a vehicle from the environment, that can
      be done by calling:

        >>> from flow.envs.base_env import Env
        >>> env = Env(...)
        >>> veh_id = "test_car"  # name of the vehicle
        >>> speed = env.k.vehicle.get_speed(veh_id)

    All methods in this class are abstract, and must be filled in by the child
    vehicle kernel of separate simulators.
    """

    def __init__(self,
                 master_kernel,
                 sim_params):
        """See parent class."""
        KernelVehicle.__init__(self, master_kernel, sim_params)

        self.__ids = []  # ids of all vehicles
        self.__human_ids = []  # ids of human-driven vehicles
        self.__controlled_ids = []  # ids of flow-controlled vehicles
        self.__controlled_lc_ids = []  # ids of flow lc-controlled vehicles
        self.__rl_ids = []  # ids of rl-controlled vehicles
        self.__observed_ids = []  # ids of the observed vehicles

        # vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        # Ordered dictionary used to keep neural net inputs in order
        self.__vehicles = collections.OrderedDict()

        # total number of vehicles in the network
        self.num_vehicles = 0
        # number of rl vehicles in the network
        self.num_rl_vehicles = 0

        # contains the parameters associated with each type of vehicle
        self.type_parameters = {}

        # list of vehicle ids located in each edge in the network
        self._ids_by_edge = dict()

        # number of vehicles that entered the network for every time-step
        self._num_departed = []
        self._departed_ids = []

        # number of vehicles to exit the network for every time-step
        self._num_arrived = []
        self._arrived_ids = []

        # contains conversion from Flow-ID to Aimsun-ID
        self._id_aimsun2flow = {}
        self._id_flow2aimsun = {}

        # contains conversion from Flow-type to Aimsun-type
        self._type_aimsun2flow = {}
        self._type_flow2aimsun = {}

        # number of vehicles of each type
        self.num_type = {}

    def initialize(self, vehicles):
        """

        :param vehicles:
        :return:
        """
        self.type_parameters = vehicles.type_parameters
        self.num_vehicles = 0
        self.num_rl_vehicles = 0

    def pass_api(self, kernel_api):
        """See parent class.

        This is also used to store conversions from Aimsun type to Flow type,
        and vise versa.
        """
        self.kernel_api = kernel_api

        self._type_aimsun2flow = {}
        self._type_flow2aimsun = {}
        for flow_type in self.type_parameters:
            # initialize the dictionary of number of types with zeros for each
            # type
            self.num_type[flow_type] = 0

            # create the dictionaries that are used to convert Aimsun vehicle
            # types to Flow vehicle types and vice versa
            aimsun_type = self.kernel_api.get_vehicle_type_id(flow_type)
            self._type_aimsun2flow[aimsun_type] = flow_type
            self._type_flow2aimsun[flow_type] = aimsun_type

    ###########################################################################
    #               Methods for interacting with the simulator                #
    ###########################################################################

    def update(self, reset):
        """See parent class.

        This is used to store an updated vehicle information object.
        """
        # collect the entered and exited vehicle_ids
        added_vehicles = self.kernel_api.get_entered_ids()
        exited_vehicles = self.kernel_api.get_exited_ids()

        # add the new vehicles
        for aimsun_id in added_vehicles:
            self._add_departed(aimsun_id)

        # remove the exited vehicles
        if not reset:
            for veh_id in exited_vehicles:
                self.remove(veh_id)

        for veh_id in self.__ids:
            aimsun_id = self._id_flow2aimsun[veh_id]

            # update the vehicle's tracking information
            tracking_info = self.__vehicles[veh_id]['tracking_info']
            (tracking_info.CurrentPos,
             tracking_info.distance2End,
             tracking_info.xCurrentPos,
             tracking_info.yCurrentPos,
             tracking_info.zCurrentPos,
             tracking_info.CurrentSpeed,
             tracking_info.TotalDistance,
             tracking_info.SectionEntranceT,
             tracking_info.CurrentStopTime,
             tracking_info.stopped,
             tracking_info.idSection,
             tracking_info.segment,
             tracking_info.numberLane,
             tracking_info.idJunction,
             tracking_info.idSectionFrom,
             tracking_info.idLaneFrom,
             tracking_info.idSectionTo,
             tracking_info.idLaneTo) = \
                self.kernel_api.get_vehicle_tracking_info(aimsun_id)

            # get the leader and followers
            lead_id = self.kernel_api.get_vehicle_leader(aimsun_id)
            if lead_id < -1:
                self.__vehicles[veh_id]['leader'] = None
            else:
                lead_id = self._id_aimsun2flow[lead_id]
                self.__vehicles[veh_id]['leader'] = lead_id
                self.__vehicles[lead_id]['follower'] = veh_id

        # update the headways and tailways
        for veh_id in self.__ids:
            aimsun_id = self._id_flow2aimsun[veh_id]
            if self.__vehicles[veh_id]['leader'] is None:
                self.__vehicles[veh_id]['headway'] = 1000
            else:
                self.__vehicles[veh_id]['headway'] = \
                    self.kernel_api.get_vehicle_headway(aimsun_id)

        # update the tailways of all vehicles TODO
        # for veh_id in self.__ids:

    def _add_departed(self, aimsun_id):
        """See parent class."""
        # get vehicle information from API
        static_inf_veh = self.kernel_api.get_vehicle_static_info(aimsun_id)

        # get the vehicle's type
        aimsun_type = static_inf_veh.type

        # convert the type to a Flow-specific type
        type_id = self._type_aimsun2flow[aimsun_type]

        # get the vehicle ID, or create a new vehicle ID if one doesn't exist
        # for the vehicle
        if aimsun_id not in self._id_aimsun2flow.keys():
            # get a new name for this vehicle
            veh_id = '{}_{}'.format(type_id, self.num_type[type_id])
            self.num_type[type_id] += 1
            self.__ids.append(veh_id)
            self.__vehicles[veh_id] = {}
            # set the Aimsun/Flow vehicle ID converters
            self._id_aimsun2flow[aimsun_id] = veh_id
            self._id_flow2aimsun[veh_id] = aimsun_id
        else:
            veh_id = self._id_aimsun2flow[aimsun_id]

        # store the static info
        self.__vehicles[veh_id]["static_info"] = static_inf_veh

        # store an empty tracking info object
        self.__vehicles[veh_id]['tracking_info'] = InfVeh()

        # specify the acceleration controller class
        accel_controller = \
            self.type_parameters[type_id]["acceleration_controller"]
        car_following_params = \
            self.type_parameters[type_id]["car_following_params"]
        self.__vehicles[veh_id]["acc_controller"] = \
            accel_controller[0](veh_id,
                                car_following_params=car_following_params,
                                **accel_controller[1])

        # specify the lane-changing controller class
        lc_controller = \
            self.type_parameters[type_id]["lane_change_controller"]
        self.__vehicles[veh_id]["lane_changer"] = \
            lc_controller[0](veh_id=veh_id, **lc_controller[1])

        # specify the routing controller class
        rt_controller = self.type_parameters[type_id]["routing_controller"]
        if rt_controller is not None:
            self.__vehicles[veh_id]["router"] = \
                rt_controller[0](veh_id=veh_id, router_params=rt_controller[1])
        else:
            self.__vehicles[veh_id]["router"] = None

        # add the vehicle's id to the list of vehicle ids
        if accel_controller[0] == RLController:
            self.__rl_ids.append(veh_id)
            self.num_rl_vehicles += 1
        else:
            self.__human_ids.append(veh_id)
            if accel_controller[0] != SimCarFollowingController:
                self.__controlled_ids.append(veh_id)
            if lc_controller[0] != SimLaneChangeController:
                self.__controlled_lc_ids.append(veh_id)

        # set the "last_lc" parameter of the vehicle
        self.__vehicles[veh_id]["last_lc"] = -float("inf")

        # make sure that the order of rl_ids is kept sorted
        self.__rl_ids.sort()

    def add(self, veh_id, type_id, edge, pos, lane, speed):
        """See parent class."""
        self.num_vehicles += 1
        self.__ids.append(veh_id)
        self.__vehicles[veh_id] = {}

        # add vehicle in Aimsun
        # negative one means the first feasible turn TODO get route
        next_section = -1
        aimsun_id = self.kernel_api.add_vehicle(
            edge=self.master_kernel.scenario.aimsun_edge_name(edge),
            lane=lane,
            type_id=self._type_flow2aimsun[type_id],
            pos=pos,
            speed=speed,
            next_section=next_section)

        # set the Aimsun/Flow vehicle ID converters
        self._id_aimsun2flow[aimsun_id] = veh_id
        self._id_flow2aimsun[veh_id] = aimsun_id

        # increment the number of vehicles of this type
        self.num_type[type_id] += 1

    def remove(self, veh_id):
        """See parent class."""
        try:
            aimsun_id = deepcopy(self._id_flow2aimsun[veh_id])
            self.kernel_api.remove_vehicle(aimsun_id)

            # remove from the vehicles kernel
            del self.__vehicles[veh_id]
            del self._id_aimsun2flow[aimsun_id]
            del self._id_flow2aimsun[veh_id]
            self.__ids.remove(veh_id)
            self.num_vehicles -= 1

            # remove it from all other ids (if it is there)
            if veh_id in self.__human_ids:
                self.__human_ids.remove(veh_id)
                if veh_id in self.__controlled_ids:
                    self.__controlled_ids.remove(veh_id)
                if veh_id in self.__controlled_lc_ids:
                    self.__controlled_lc_ids.remove(veh_id)
            else:
                self.__rl_ids.remove(veh_id)
                self.num_rl_vehicles -= 1
        except (KeyError, ValueError):
            print("Invalid vehicle ID to be removed")

        # make sure that the rl ids remain sorted
        self.__rl_ids.sort()

    def apply_acceleration(self, veh_ids, acc):
        """See parent class."""
        for i, veh_id in enumerate(veh_ids):
            if acc[i] is not None:
                this_vel = self.get_speed(veh_id)
                next_vel = max([this_vel + acc[i] * self.sim_step, 0])
                aimsun_id = self._id_flow2aimsun[veh_id]
                self.kernel_api.set_speed(aimsun_id, next_vel)

    def apply_lane_change(self, veh_ids, direction):
        """Apply an instantaneous lane-change to a set of vehicles.

        This method also prevents vehicles from moving to lanes that do not
        exist, and set the "last_lc" variable for RL vehicles that lane changed
        to match the current time step, in order to assist in maintaining a
        lane change duration for these vehicles.

        Parameters
        ----------
        veh_ids : list of str
            list of vehicle identifiers
        direction : list of {-2, -1, 0, 1}
            -2: reset, gives back the control to the default simulation model
            -1: lane change to the right
             0: no lane change
             1: lane change to the left

        Raises
        ------
        ValueError
            If any of the direction values are not -2, -1, 0, or 1.
        """
        # if any of the directions are not -1, 0, or 1, raise a ValueError
        if any(d not in [-2, -1, 0, 1] for d in direction):
            raise ValueError(
                "Direction values for lane changes may only be: -2, -1, 0, \
                or 1.")

        for i, veh_id in enumerate(veh_ids):
            # check for no lane change
            if direction[i] == 0:
                continue

            # compute the target lane, and clip it so vehicle don't try to lane
            # change out of range
            this_lane = self.get_lane(veh_id)
            this_edge = self.get_edge(veh_id)
            target_lane = min(
                max(this_lane + direction[i], 0),
                self.master_kernel.scenario.num_lanes(this_edge) - 1)

            # perform the requested lane action action in Aimsun
            if target_lane != this_lane:
                aimsun_id = self._id_flow2aimsun[veh_id]
                self.kernel_api.apply_lane_change(aimsun_id, int(target_lane))

                if veh_id in self.get_rl_ids():
                    self.prev_last_lc[veh_id] = \
                        self.__vehicles[veh_id]["last_lc"]

    def choose_routes(self, veh_ids, route_choices):
        """Update the route choice of vehicles in the network.

        Parameters
        ----------
        veh_ids : list
            list of vehicle identifiers
        route_choices : numpy ndarray or list of floats
            list of edges the vehicle wishes to traverse, starting with the
            edge the vehicle is currently on. If a value of None is provided,
            the vehicle does not update its route
        """
        pass  # FIXME
        # for i, veh_id in enumerate(veh_ids):
        #     if route_choices[i] is not None:
        #         aimsun_id = self._id_flow2aimsun[veh_id]
        #         size_next_sections = len(route_choices[i])
        #         self.kernel_api.AKIVehTrackedModifyNextSections(
        #             aimsun_id, size_next_sections, route_choices[i])

    ###########################################################################
    # Methods to visually distinguish vehicles by {RL, observed, unobserved}  #
    ###########################################################################

    def update_vehicle_colors(self):
        """Modify the color of vehicles if rendering is active."""
        # color rl vehicles red
        for veh_id in self.get_rl_ids():
            aimsun_id = self._id_flow2aimsun[veh_id]
            self.kernel_api.set_color(veh_id=aimsun_id, color=RED)

        # observed human-driven vehicles are cyan and unobserved are white
        for veh_id in self.get_human_ids():
            aimsun_id = self._id_flow2aimsun[veh_id]
            color = CYAN if veh_id in self.get_observed_ids() else WHITE
            self.kernel_api.set_color(veh_id=aimsun_id, color=color)

        # clear the list of observed vehicles
        for veh_id in self.get_observed_ids():
            self.remove_observed(veh_id)

    def set_observed(self, veh_id):
        """Add a vehicle to the list of observed vehicles."""
        if veh_id not in self.__observed_ids:
            self.__observed_ids.append(veh_id)

    def remove_observed(self, veh_id):
        """Remove a vehicle from the list of observed vehicles."""
        if veh_id in self.__observed_ids:
            self.__observed_ids.remove(veh_id)

    def get_observed_ids(self):
        """Return the list of observed vehicles."""
        return self.__observed_ids

    ###########################################################################
    #                        State acquisition methods                        #
    ###########################################################################

    def get_ids(self):
        """See parent class."""
        return self.__ids

    def get_human_ids(self):
        """See parent class."""
        return self.__human_ids

    def get_controlled_ids(self):
        """See parent class."""
        return self.__controlled_ids

    def get_controlled_lc_ids(self):
        """See parent class."""
        return self.__controlled_lc_ids

    def get_rl_ids(self):
        """See parent class."""
        return self.__rl_ids

    def get_ids_by_edge(self, edges):
        """See parent class."""
        if isinstance(edges, (list, np.ndarray)):
            return sum([self.get_ids_by_edge(edge) for edge in edges], [])
        return [veh for veh in self.__ids if self.get_edge(veh) == edges]

    def get_inflow_rate(self, time_span):
        """See parent class."""
        if len(self._num_departed) == 0:
            return 0
        num_inflow = self._num_departed[-int(time_span / self.sim_step):]
        return 3600 * sum(num_inflow) / (len(num_inflow) * self.sim_step)

    def get_outflow_rate(self, time_span):
        """See parent class."""
        if len(self._num_arrived) == 0:
            return 0
        num_outflow = self._num_arrived[-int(time_span / self.sim_step):]
        return 3600 * sum(num_outflow) / (len(num_outflow) * self.sim_step)

    def get_num_arrived(self):
        """See parent class."""
        if len(self._num_arrived) > 0:
            return self._num_arrived[-1]
        else:
            return 0

    def get_type(self, veh_id):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_type(veh) for veh in veh_id]
        aimsun_type = self.__vehicles[veh_id]['static_info'].type
        return self._type_aimsun2flow[aimsun_type]

    def get_speed(self, veh_id, error=-1001):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_speed(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['tracking_info'].CurrentSpeed / 3.6

    def get_position(self, veh_id, error=-1001):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_position(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['tracking_info'].CurrentPos

    def get_position_world(self, veh_id, error=-1001):
        """Return the position of the vehicle relative to its current edge.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float,
            The distance from the beginning of the section
        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_position_world(veh, error) for veh in veh_id]
        x_pos = self.__vehicles[veh_id]['tracking_info'].xCurrentPos
        y_pos = self.__vehicles[veh_id]['tracking_info'].yCurrentPos
        z_pos = self.__vehicles[veh_id]['tracking_info'].zCurrentPos
        return [x_pos, y_pos, z_pos]

    def get_edge(self, veh_id, error=""):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_edge(veh, error) for veh in veh_id]
        edge_aimsun_id = self.__vehicles[veh_id]['tracking_info'].idSection
        if edge_aimsun_id < 0:
            # TODO: add from and to lanes in junctions
            from_edge = self.master_kernel.scenario.flow_edge_name(
                self.__vehicles[veh_id]['tracking_info'].idSectionFrom)
            to_edge = self.master_kernel.scenario.flow_edge_name(
                self.__vehicles[veh_id]['tracking_info'].idSectionTo)
            return '{}_to_{}'.format(from_edge, to_edge)
        else:
            return self.master_kernel.scenario.flow_edge_name(edge_aimsun_id)

    def get_lane(self, veh_id, error=-1001):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['tracking_info'].numberLane

    def get_route(self, veh_id, error=None):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_route(veh) for veh in veh_id]
        return []  # FIXME
        # aimsun_id = self._id_flow2aimsun[veh_id]
        # num_secs = self.kernel_api.AKIVehTrackedGetNbSectionsVehiclePath(
        #     aimsun_id)
        # return self.kernel_api.AKIVehTrackedGetIdSectionVehiclePath(
        #     aimsun_id, num_secs)

    def get_length(self, veh_id, error=-1001):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_length(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['static_info'].length

    def get_leader(self, veh_id, error=""):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_leader(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['leader'] or error

    def get_follower(self, veh_id, error=""):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_follower(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['follower']

    def get_headway(self, veh_id, error=-1001):
        """See parent class."""
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_headway(veh, error) for veh in veh_id]
        return self.__vehicles[veh_id]['headway']

    def get_last_lc(self, veh_id, error=-1001):
        """See parent class."""
        return self.__vehicles[veh_id]["last_lc"]

    def get_acc_controller(self, veh_id, error=None):
        """See parent class."""
        return self.__vehicles[veh_id]["acc_controller"]

    def get_lane_changing_controller(self, veh_id, error=None):
        """See parent class."""
        return self.__vehicles[veh_id]["lane_changer"]

    def get_routing_controller(self, veh_id, error=None):
        """See parent class."""
        return self.__vehicles[veh_id]["router"]

    def get_x_by_id(self, veh_id):
        """See parent class."""
        return self.master_kernel.scenario.get_x(self.get_edge(veh_id),
                                                 self.get_position(veh_id))

    def set_lane_headways(self, veh_id, lane_headways):
        """See parent class."""
        raise NotImplementedError

    def get_lane_headways(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError

    def set_lane_leaders(self, veh_id, lane_leaders):
        """See parent class."""
        raise NotImplementedError

    def get_lane_leaders(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError

    def set_lane_tailways(self, veh_id, lane_tailways):
        """See parent class."""
        raise NotImplementedError

    def get_lane_tailways(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError

    def set_lane_followers(self, veh_id, lane_followers):
        """See parent class."""
        raise NotImplementedError

    def get_lane_followers(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError

    def get_lane_followers_speed(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError

    def get_lane_leaders_speed(self, veh_id, error=None):
        """See parent class."""
        raise NotImplementedError
