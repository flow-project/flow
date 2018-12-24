"""Script containing the base vehicle kernel class."""
from flow.core.kernel.vehicle.base import KernelVehicle
import collections
import numpy as np
import itertools
from flow.controllers.car_following_models import SimCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController


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
        >>> veh_id = "test_car"  # name of the vehicle #TODO veh_ids should be numbers
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

        # create a sumo_observations variable that will carry all information
        # on the state of the vehicles for a given time step
        self.__sumo_obs = {}

        # total number of vehicles in the network
        self.num_vehicles = 0
        # number of rl vehicles in the network
        self.num_rl_vehicles = 0

        # contains the parameters associated with each type of vehicle
        self.type_parameters = {}

        # contain the minGap attribute of each type of vehicle
        self.minGap = {}

        # list of vehicle ids located in each edge in the network
        self._ids_by_edge = dict()

        # number of vehicles that entered the network for every time-step
        self._num_departed = []
        self._departed_ids = []

        # number of vehicles to exit the network for every time-step
        self._num_arrived = []
        self._arrived_ids = []

        # contains the Flow IDs of all vehicle based on their Aimsun IDs
        self._aimsun_to_veh_id = {}

    def initialize(self, vehicles):
        """

        :param vehicles:
        :return:
        """
        self.type_parameters = vehicles.type_parameters
        self.minGap = vehicles.minGap
        self.num_vehicles = 0
        self.num_rl_vehicles = 0

    def pass_api(self, kernel_api):
        """See parent class."""
        self.kernel_api = kernel_api

    ###########################################################################
    #               Methods for interacting with the simulator                #
    ###########################################################################

    def update(self, reset):
        """See parent class.

        This is used to store an updated vehicle information object.
        """
        # update the vehicle_ids object TODO

        # update all vehicles' tracking information
        for veh_id in self.__ids:
            self.__vehicles['tracking_info'] = \
                self.kernel_api.get_vehicle_tracking_info(veh_id)

    def add(self, veh_id, type_id, edge, pos, lane, speed):
        """See parent class."""
        self.num_vehicles += 1
        self.__ids.append(veh_id)
        self.__vehicles[veh_id] = {}

        # specify the type
        self.__vehicles[veh_id]["type"] = type_id

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

        # add vehicle in Aimsun
        # negative one means the first feasible turn TODO get route
        next_section = -1
        # TODO: edge in aimsun
        # TODO: type as aimsun wants it (i.e. int)
        aimsun_id = self.kernel_api.add_vehicle(
            edge, lane, type_id, pos, speed, next_section
        )

        # get vehicle information from API
        static_inf_veh = self.kernel_api.get_vehicle_static_info(aimsun_id)
        self.__vehicles[veh_id]["static_info"] = static_inf_veh

        # set Aimsun ID
        self.__vehicles[veh_id]["aimsun_id"] = aimsun_id

        # set veh_id to aimsun id
        self._aimsun_to_veh_id[aimsun_id] = veh_id

        # set the absolute position of the vehicle
        self.__vehicles[veh_id]["absolute_position"] = 0

        # set the "last_lc" parameter of the vehicle
        self.__vehicles[veh_id]["last_lc"] = -float("inf")

        # make sure that the order of rl_ids is kept sorted
        self.__rl_ids.sort()

    def remove(self, veh_id):
        """Remove a vehicle.

        This method removes all traces of the vehicle from the vehicles kernel
        and all valid ID lists, and decrements the total number of vehicles in
        this class.

        In addition, if the vehicle is still in the network, this method calls
        the necessary simulator-specific commands to remove it.

        Parameters
        ----------
        veh_id : str
            unique identifier of the vehicle to be removed
        """
        try:
            aimsun_id = self.__vehicles[veh_id]["aimsun_id"]
            self.kernel_api.remove_vehicle(aimsun_id)
        except ValueError:
            print("Invalid vehicle ID to be removed")

        # remove from the vehicles kernel
        del self.__vehicles[veh_id]
        del self.__sumo_obs[veh_id]
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

        # make sure that the rl ids remain sorted
        self.__rl_ids.sort()

    def apply_acceleration(self, veh_ids, acc):
        """Apply the acceleration requested by a vehicle in the simulator.

        Parameters
        ----------
        veh_ids : list of str
            list of vehicle identifiers
        acc : numpy ndarray or list of float
            requested accelerations from the vehicles
        """
        for i, veh_id in enumerate(veh_ids):
            if acc[i] is not None:
                this_vel = self.get_speed(veh_id)
                next_vel = max([this_vel + acc[i] * self.sim_step, 0])
                aimsun_id = self.__vehicles[veh_id]["aimsun_id"]
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
            If any of the direction values are not -1, 0, or 1.
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
                aimsun_id = self.__vehicles[veh_id]["aimsun_id"]
                self.kernel_api.apply_lane_change(aimsun_id, int(target_lane))

                if veh_id in self.get_rl_ids():
                    self.prev_last_lc[veh_id] = \
                        self.__vehicles[veh_id]["last_lc"]

                # get simulation time
                time = self.kernel_api.AKIGetCurrentSimulationTime()
                self.__vehicles[veh_id]["last_lc"] = time  # TODO this is missing in Tracy

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
        for i, veh_id in enumerate(veh_ids):
            if route_choices[i] is not None:
                aimsun_id = self.__vehicles[veh_id]["aimsun_id"]
                size_next_sections = len(route_choices[i])
                self.kernel_api.AKIVehTrackedModifyNextSections(
                    aimsun_id, size_next_sections, route_choices[i])

    ###########################################################################
    # Methods to visually distinguish vehicles by {RL, observed, unobserved}  #
    ###########################################################################

    # FIXME: maybe add later?
    def update_vehicle_colors(self):
        """Modify the color of vehicles if rendering is active."""
        pass

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
        veh_ids = []
        num_vehs = self.kernel_api.AKIVehStateGetNbVehiclesSection(
            edges["aimsun_id"], True)
        for j in range(num_vehs):
            inf_veh = self.kernel_api.AKIVehStateGetVehicleInfSection(
                edge["aimsun_id"], j)
            aimsun_id = inf_veh.idVeh
            veh_ids.append(self._aimsun_to_veh_id[aimsun_id])
        return veh_ids

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

    def get_speed(self, veh_id, error=-1001):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        speeds = []
        for veh in veh_id:
            speed = self.__vehicles[veh]["tracking_info"].CurrentSpeed
            speeds.append(speed)
        return speeds

    def get_absolute_position(self, veh_id, error=-1001):
        """See parent class."""
        raise NotImplementedError  # TODO check

    def get_position(self, veh_id, error=-1001):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        positions = []
        for veh in veh_id:
            pos = self.__vehicles[veh]['tracking_info'].CurrentPos
            positions.append(pos)
        return positions

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
        positions = []
        for veh in veh_id:
            x_pos = self.__vehicles[veh]['tracking_info'].xCurrentPos
            y_pos = self.__vehicles[veh]['tracking_info'].yCurrentPos
            z_pos = self.__vehicles[veh]['tracking_info'].zCurrentPos
            positions.append([x_pos, y_pos, z_pos])
        return positions

    def get_edge(self, veh_id, error=""):
        """See parent class."""
        edges = []
        for veh in veh_id:
            aimsun_id = self.__vehicles[veh]["aimsun_id"]
            inf_veh = self.kernel_api.AKIVehTrackedGetInf(aimsun_id)
            edge_aimsun_id = inf_veh.idSection
            edges.append(edge for edge in
                         self.master_kernel.scenario.get_edge_list()
                         if edge["aimsun_id"] == edge_aimsun_id)
        return edges

    def get_lane(self, veh_id, error=-1001):
        """See parent class."""
        lanes = []
        for veh in veh_id:
            lane = self.__vehicles[veh]['tracking_info'].numberLane
            lanes.append(lane)
        return lanes

    def get_route(self, veh_id, error=list()):
        """See parent class."""
        routes = []
        for veh in veh_id:
            aimsun_id = self.__vehicles[veh]["aimsun_id"]
            num_secs = self.kernel_api.AKIVehTrackedGetNbSectionsVehiclePath(
                aimsun_id)
            route = self.kernel_api.AKIVehTrackedGetIdSectionVehiclePath(
                aimsun_id, num_secs)
            routes.append(route)
        return routes

    def get_length(self, veh_id, error=-1001):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        lengths = []
        for veh in veh_id:
            lengths.append(self.__vehicles[veh]["length"])  # TODO double check
        return lengths

    def get_leader(self, veh_id, error=""):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        leaders = []
        for veh in veh_id:
            aimsun_id = self.__vehicles[veh]["aimsun_id"]
            leader_id = self.kernel_api.AKIVehGetLeaderId(aimsun_id)
            leaders.append(self.__vehicles[leader_id]["vehicle_id"])
        return leaders

    def get_follower(self, veh_id, error=""):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        followers = []
        for veh in veh_id:
            aimsun_id = self.__vehicles[veh]["aimsun_id"]
            follower_id = self.kernel_api.AKIVehGetFollowerId(aimsun_id)
            followers.append(self.__vehicles[follower_id]["vehicle_id"])
        return followers

    def get_headway(self, veh_id, error=-1001):
        """See parent class."""
        # FIXME: do it the way we do, in case veh_id is not a list
        headways = []
        for veh in veh_id:
            leader_id = self.get_leader(veh, error)
            if self.get_edge(leader_id, error) is self.get_edge(veh, error):
                gap = self.get_position(leader_id, error) \
                      - self.get_position(veh) \
                      - self.get_length(leader_id, error)
                headways.append(gap)
            else:
                # assume Euclidean distance
                leader_pos = self.get_position_world(leader_id, error)
                veh_pos = self.get_position_world(veh, error)
                dist = np.linalg.norm(
                    np.array(leader_pos)-np.array(veh_pos))
                gap = dist - self.get_length(leader_id, error)
                headways.append(gap)
        # TODO check this for the better way
        # if inf_leader.idSection != inf_veh.idSection:
        #     num_secs = self.kernel_api.AKIInfNetGetShortestPathNbSections(
        #         inf_veh.idSection, inf_leader.idSection, sectionCostColumn
        #     )
        #
        #     if num_secs > 0:
        #         path = intArray(num_secs)
        #         result = self.kernel_api.AKIInfNetGetShortestPath(
        #             inf_veh.idSection, inf_leader.idSection, sectionCostColumn,
        #             path
        #         )
        #         for index in range(0, num_secs):
        #             AKIPrintString( "%d"%path[index] )
        #             #dist_in_between +=
        #
        # gap = inf_leader.CurrentPos - lead_veh_length - inf_veh.CurrentPos \
        #     + dist_in_between
        return headways

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
        # FIXME
        raise NotImplementedError

    def set_lane_headways(self, veh_id, lane_headways):
        """Set the lane headways of the specified vehicle."""
        self.__vehicles[veh_id]["lane_headways"] = lane_headways

    def get_lane_headways(self, veh_id, error=list()):
        """Return the lane headways of the specified vehicles.

        This includes the headways between the specified vehicle and the
        vehicle immediately ahead of it in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_headways(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_headways", error)

    def set_lane_leaders(self, veh_id, lane_leaders):
        """Set the lane leaders of the specified vehicle."""
        self.__vehicles[veh_id]["lane_leaders"] = lane_leaders

    def get_lane_leaders(self, veh_id, error=list()):
        """Return the leaders for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        lis of str
        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_leaders(vehID, error) for vehID in veh_id]
        return self.__vehicles[veh_id]["lane_leaders"]

    def set_lane_tailways(self, veh_id, lane_tailways):
        """Set the lane tailways of the specified vehicle."""
        self.__vehicles[veh_id]["lane_tailways"] = lane_tailways

    def get_lane_tailways(self, veh_id, error=list()):
        """Return the lane tailways of the specified vehicle.

        This includes the headways between the specified vehicle and the
        vehicle immediately behind it in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of float
        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_tailways(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_tailways", error)

    def set_lane_followers(self, veh_id, lane_followers):
        """Set the lane followers of the specified vehicle."""
        self.__vehicles[veh_id]["lane_followers"] = lane_followers

    def get_lane_followers(self, veh_id, error=list()):
        """Return the followers for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : list, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list of str
        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_followers(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_followers", error)

    def _multi_lane_headways(self):
        """Compute multi-lane data for all vehicles.

        This includes the lane leaders/followers/headways/tailways for all
        vehicles in the network.
        """
        edge_list = self.master_kernel.scenario.get_edge_list()
        junction_list = self.master_kernel.scenario.get_junction_list()
        tot_list = edge_list + junction_list
        num_edges = (len(self.master_kernel.scenario.get_edge_list()) + len(
            self.master_kernel.scenario.get_junction_list()))

        # maximum number of lanes in the network
        max_lanes = max([self.master_kernel.scenario.num_lanes(edge_id)
                         for edge_id in tot_list])

        # Key = edge id
        # Element = list, with the ith element containing tuples with the name
        #           and position of all vehicles in lane i
        edge_dict = dict.fromkeys(tot_list)

        # add the vehicles to the edge_dict element
        for veh_id in self.get_ids():
            edge = self.get_edge(veh_id)
            lane = self.get_lane(veh_id)
            pos = self.get_position(veh_id)
            if edge:
                if edge_dict[edge] is None:
                    edge_dict[edge] = [[] for _ in range(max_lanes)]
                edge_dict[edge][lane].append((veh_id, pos))

        # sort all lanes in each edge by position
        for edge in tot_list:
            if edge_dict[edge] is None:
                del edge_dict[edge]
            else:
                for lane in range(max_lanes):
                    edge_dict[edge][lane].sort(key=lambda x: x[1])

        for veh_id in self.get_rl_ids():
            # collect the lane leaders, followers, headways, and tailways for
            # each vehicle
            edge = self.get_edge(veh_id)
            if edge:
                headways, tailways, leaders, followers = \
                    self._multi_lane_headways_util(veh_id, edge_dict,
                                                   num_edges)

                # add the above values to the vehicles class
                self.set_lane_headways(veh_id, headways)
                self.set_lane_tailways(veh_id, tailways)
                self.set_lane_leaders(veh_id, leaders)
                self.set_lane_followers(veh_id, followers)

        self._ids_by_edge = dict().fromkeys(edge_list)

        for edge_id in edge_dict:
            edges = list(itertools.chain.from_iterable(edge_dict[edge_id]))
            # check for edges with no vehicles
            if len(edges) > 0:
                edges, _ = zip(*edges)
                self._ids_by_edge[edge_id] = list(edges)
            else:
                self._ids_by_edge[edge_id] = []

    def _multi_lane_headways_util(self, veh_id, edge_dict, num_edges):
        """Compute multi-lane data for the specified vehicle.

        Parameters
        ----------
        veh_id : str
            name of the vehicle
        edge_dict : dict < list<tuple> >
            Key = Edge name
                Index = lane index
                Element = list sorted by position of (vehicle id, position)

        Returns
        -------
        headway : list<float>
            Index = lane index
            Element = headway at this lane
        tailway : list<float>
            Index = lane index
            Element = tailway at this lane
        leader : list<str>
            Index = lane index
            Element = leader at this lane
        follower : list<str>
            Index = lane index
            Element = follower at this lane
        """
        this_pos = self.get_position(veh_id)
        this_edge = self.get_edge(veh_id)
        this_lane = self.get_lane(veh_id)
        num_lanes = self.master_kernel.scenario.num_lanes(this_edge)

        # set default values for all output values
        headway = [1000] * num_lanes
        tailway = [1000] * num_lanes
        leader = [""] * num_lanes
        follower = [""] * num_lanes

        for lane in range(num_lanes):
            # check the vehicle's current  edge for lane leaders and followers
            if len(edge_dict[this_edge][lane]) > 0:
                ids, positions = zip(*edge_dict[this_edge][lane])
                ids = list(ids)
                positions = list(positions)
                index = bisect_left(positions, this_pos)

                # if you are at the end or the front of the edge, the lane
                # leader is in the edges in front of you
                if (lane == this_lane and index < len(positions) - 1) \
                        or (lane != this_lane and index < len(positions)):
                    # check if the index does not correspond to the current
                    # vehicle
                    if ids[index] == veh_id:
                        leader[lane] = ids[index + 1]
                        headway[lane] = (positions[index + 1] - this_pos -
                                         self.get_length(leader[lane]))
                    else:
                        leader[lane] = ids[index]
                        headway[lane] = (positions[index] - this_pos
                                         - self.get_length(leader[lane]))

                # you are in the back of the queue, the lane follower is in the
                # edges behind you
                if index > 0:
                    follower[lane] = ids[index - 1]
                    tailway[lane] = (this_pos - positions[index - 1]
                                     - self.get_length(veh_id))

            # if lane leader not found, check next edges
            if leader[lane] == "":
                headway[lane], leader[lane] = self._next_edge_leaders(
                    veh_id, edge_dict, lane, num_edges)

            # if lane follower not found, check previous edges
            if follower[lane] == "":
                tailway[lane], follower[lane] = self._prev_edge_followers(
                    veh_id, edge_dict, lane, num_edges)

        return headway, tailway, leader, follower

    def _next_edge_leaders(self, veh_id, edge_dict, lane, num_edges):
        """Search for leaders in the next edge.

        Looks to the edges/junctions in front of the vehicle's current edge
        for potential leaders. This is currently done by only looking one
        edge/junction forwards.

        Returns
        -------
        headway : float
            lane headway for the specified lane
        leader : str
            lane leader for the specified lane
        """
        pos = self.get_position(veh_id)
        edge = self.get_edge(veh_id)

        headway = 1000  # env.scenario.length
        leader = ""
        add_length = 0  # length increment in headway

        for _ in range(num_edges):
            # break if there are no edge/lane pairs behind the current one
            if len(self.master_kernel.scenario.next_edge(edge, lane)) == 0:
                break

            add_length += self.master_kernel.scenario.edge_length(edge)
            edge, lane = self.master_kernel.scenario.next_edge(edge, lane)[0]

            try:
                if len(edge_dict[edge][lane]) > 0:
                    leader = edge_dict[edge][lane][0][0]
                    headway = edge_dict[edge][lane][0][1] - pos + add_length \
                        - self.get_length(leader)
            except KeyError:
                # current edge has no vehicles, so move on
                continue

            # stop if a lane follower is found
            if leader != "":
                break

        return headway, leader

    def _prev_edge_followers(self, veh_id, edge_dict, lane, num_edges):
        """Search for followers in the previous edge.

        Looks to the edges/junctions behind the vehicle's current edge for
        potential followers. This is currently done by only looking one
        edge/junction backwards.

        Returns
        -------
        tailway : float
            lane tailway for the specified lane
        follower : str
            lane follower for the specified lane
        """
        pos = self.get_position(veh_id)
        edge = self.get_edge(veh_id)

        tailway = 1000  # env.scenario.length
        follower = ""
        add_length = 0  # length increment in headway

        for _ in range(num_edges):
            # break if there are no edge/lane pairs behind the current one
            if len(self.master_kernel.scenario.prev_edge(edge, lane)) == 0:
                break

            edge, lane = self.master_kernel.scenario.prev_edge(edge, lane)[0]
            add_length += self.master_kernel.scenario.edge_length(edge)

            try:
                if len(edge_dict[edge][lane]) > 0:
                    tailway = pos - edge_dict[edge][lane][-1][1] + add_length \
                              - self.get_length(veh_id)
                    follower = edge_dict[edge][lane][-1][0]
            except KeyError:
                # current edge has no vehicles, so move on
                continue

            # stop if a lane follower is found
            if follower != "":
                break

        return tailway, follower
