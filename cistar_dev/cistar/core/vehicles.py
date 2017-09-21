from cistar.controllers.base_controller import SumoController
from cistar.controllers.rlcontroller import RLController

import collections


class Vehicles:
    def __init__(self):
        """
        Base vehicle class used to describe the state of all vehicles in the
        network. State information on the vehicles for a given time step can be
        set or retreived from this class.
        """
        self.__ids = []  # used to store the ids of all vehicles
        self.__controlled_ids = []  # used to store the ids of cistar-controlled vehicles
        self.__sumo_ids = []  # used to store the ids of sumo-controlled vehicles
        self.__rl_ids = []  # used to store the ids of rllab-controlled vehicles

        # vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        # Ordered dictionary used to keep neural net inputs in order
        self.__vehicles = collections.OrderedDict()

        self.num_vehicles = 0  # total number of vehicles in the network
        self.num_rl_vehicles = 0  # number of rl vehicles in the network
        self.num_types = 0  # number of unique types of vehicles in the network
        self.types = []  # types of vehicles in the network
        self.initial_speeds = []  # speed of vehicles at the start of a rollout

    def add_vehicles(self,
                     veh_id,
                     acceleration_controller,
                     lane_change_controller=None,
                     routing_controller=None,
                     initial_speed=0,
                     num_vehicles=1):
        """
        Adds a sequence of vehicles to the list of vehicles in the network.

        Parameters
        ----------
        veh_id: str
            base vehicle ID for the vehicles (will be appended by a number)
        acceleration_controller: tup
            1st element: cistar-specified acceleration controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        lane_change_controller: tup, optional
            1st elemnt: cistar-specified lane-changer controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        routing_controller: tup, optional
            1st element: cistar-specified routing controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        initial_speed: float, optional
            initial speed of the vehicles being added (in m/s)
        num_vehicles: int, optional
            number of vehicles of this type to be added to the network
        """
        if not veh_id:
            raise ValueError("No vehicle id is specified.")

        if not acceleration_controller:
            raise ValueError("No acceleration controller is specified.")

        for i in range(num_vehicles):
            vehID = veh_id + '_%d' % i

            # add the vehicle to the list of vehicle ids
            self.__ids.append(vehID)

            self.__vehicles[vehID] = dict()

            # specify the type
            self.__vehicles[vehID]["type"] = veh_id

            # specify the acceleration controller class
            self.__vehicles[vehID]["acc_controller"] = \
                acceleration_controller[0](veh_id=vehID, **acceleration_controller[1])

            # specify the lane-changing controller class
            if lane_change_controller is not None:
                self.__vehicles[vehID]["lane_changer"] = \
                    lane_change_controller[0](veh_id=vehID, **lane_change_controller[1])
            else:
                self.__vehicles[vehID]["lane_changer"] = None

            # specify the routing controller class
            if routing_controller is not None:
                self.__vehicles[vehID]["router"] = \
                    routing_controller[0](veh_id=vehID, router_params=routing_controller[1])
            else:
                self.__vehicles[vehID]["router"] = None

            # specify the speed of vehicles at the start of a rollout
            self.__vehicles[vehID]["initial_speed"] = initial_speed

            # determine the type of vehicle, and append it to its respective
            # id list
            if acceleration_controller[0] == SumoController:
                self.__sumo_ids.append(vehID)
            elif acceleration_controller[0] == RLController:
                self.__rl_ids.append(vehID)
            else:
                self.__controlled_ids.append(vehID)

        # update the variables for the number of vehicles in the network
        self.num_vehicles = len(self.__ids)
        self.num_rl_vehicles = len(self.__rl_ids)

        # increase the number of unique types of vehicles in the network, and
        # add the type to the list of types
        self.num_types += 1
        self.types.append(veh_id)

    def set_speed(self, veh_id, speed):
        self.__vehicles[veh_id]["speed"] = speed

    def set_absolute_position(self, veh_id, absolute_position):
        self.__vehicles[veh_id]["absolute_position"] = absolute_position

    def set_position(self, veh_id, position):
        self.__vehicles[veh_id]["position"] = position

    def set_edge(self, veh_id, edge):
        self.__vehicles[veh_id]["edge"] = edge

    def set_lane(self, veh_id, lane):
        self.__vehicles[veh_id]["lane"] = lane

    def set_route(self, veh_id, route):
        self.__vehicles[veh_id]["route"] = route

    def set_leader(self, veh_id, leader):
        self.__vehicles[veh_id]["leader"] = leader

    def set_follower(self, veh_id, follower):
        self.__vehicles[veh_id]["follower"] = follower

    def set_headway(self, veh_id, headway):
        self.__vehicles[veh_id]["headway"] = headway

    def set_state(self, veh_id, state_name, state):
        """
        Generic set function. Updates the state *state_name* of the vehicle with
        id *veh_id* with the value *state*.
        """
        self.__vehicles[veh_id][state_name] = state

    def get_ids(self):
        return self.__ids

    def get_controlled_ids(self):
        return self.__controlled_ids

    def get_sumo_ids(self):
        return self.__sumo_ids

    def get_rl_ids(self):
        return self.__rl_ids

    def get_initial_speed(self, veh_id):
        return self.__vehicles[veh_id]["initial_speed"]

    def get_speed(self, veh_id="all"):
        """
        Return the speed of the specified vehicle at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["speed"] for vehID in veh_id]
        if veh_id == "all":
            return [self.__vehicles[vehID]["speed"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["speed"]

    def get_absolute_position(self, veh_id="all"):
        """
        Return the absolute position of the specified vehicle at the current
        time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["absolute_position"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["absolute_position"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["absolute_position"]

    def get_position(self, veh_id="all"):
        """
        Returns the position of the specified vehicle (relative to the current
        edge) at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["position"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["position"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["position"]

    def get_edge(self, veh_id="all"):
        """
        Returns the position of the specified vehicle (relative to the current
        edge) at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["edge"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["edge"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["edge"]

    def get_lane(self, veh_id="all"):
        """
        Returns the lane index of the specified vehicle at the current time step.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["lane"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["lane"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["lane"]

    def get_acc_controller(self, veh_id="all"):
        """
        Returns the acceleration controller of the specified vehicle.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["acc_controller"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["acc_controller"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["acc_controller"]

    def get_lane_changing_controller(self, veh_id="all"):
        """
        Returns the lane changing controller of the specified vehicle.

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["lane_changer"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["lane_changer"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["lane_changer"]

    def get_routing_controller(self, veh_id="all"):
        """
        Returns the routing controller of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["router"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["router"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["router"]

    def get_route(self, veh_id="all"):
        """
        Returns the route of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["route"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["route"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["route"]

    def get_leader(self, veh_id="all"):
        """
        Returns the leader of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["leader"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["leader"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["leader"]

    def get_follower(self, veh_id="all"):
        """
        Returns the follower of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["follower"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["follower"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["follower"]

    def get_headway(self, veh_id="all"):
        """
        Returns the headway of the specified vehicle(s).

        Accepts as input:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID]["headway"] for vehID in veh_id]
        elif veh_id == "all":
            return [self.__vehicles[vehID]["headway"] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id]["headway"]

    def get_state(self, veh_id, state_name):
        """
        Generic get function. Returns the value of *state_name* of the specified
        vehicles at the current time step.

        veh_id may be:
        - id of a specific vehicle
        - list of vehicle ids
        - "all", in which case a list of all the specified state is provided
        """
        if isinstance(veh_id, list):
            return [self.__vehicles[vehID][state_name] for vehID in veh_id]
        if veh_id == "all":
            return [self.__vehicles[vehID][state_name] for vehID in self.__ids]
        else:
            return self.__vehicles[veh_id][state_name]

    def get_full_state(self, veh_id):
        """
        Return a dict of all state variables of a specific vehicle:
        """
        return self.__vehicles[veh_id]
