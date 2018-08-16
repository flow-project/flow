from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
import collections
import logging
from bisect import bisect_left
import itertools
import numpy as np

import traci.constants as tc

from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams

SPEED_MODES = {"aggressive": 0, "no_collide": 1, "right_of_way": 25,
               "all_checks": 31}
LC_MODES = {"aggressive": 0, "no_lat_collide": 512, "strategic": 853}


class Vehicles:

    def __init__(self):
        """Base vehicle class.

        This is used to describe the state of all vehicles in the network.
        State information on the vehicles for a given time step can be set or
        retrieved from this class.
        """
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
        self.__sumo_obs = None

        self.num_vehicles = 0  # total number of vehicles in the network
        self.num_rl_vehicles = 0  # number of rl vehicles in the network
        self.num_types = 0  # number of unique types of vehicles in the network
        self.types = []  # types of vehicles in the network
        self.initial_speeds = []  # speed of vehicles at the start of a rollout

        # contains the parameters associated with each type of vehicle
        self.type_parameters = dict()

        # contain the minGap attribute of each type of vehicle
        self.minGap = dict()

        # list of vehicle ids located in each edge in the network
        self._ids_by_edge = dict()

        # number of vehicles that entered the network for every time-step
        self._num_departed = []

        # number of vehicles to exit the network for every time-step
        self._num_arrived = []

        # simulation step size
        self.sim_step = 0

        # initial state of the vehicles class, used for serialization purposes
        self.initial = []

    def add(self,
            veh_id,
            acceleration_controller=(SumoCarFollowingController, {}),
            lane_change_controller=(SumoLaneChangeController, {}),
            routing_controller=None,
            initial_speed=0,
            num_vehicles=1,
            speed_mode='right_of_way',
            lane_change_mode="no_lat_collide",
            sumo_car_following_params=None,
            sumo_lc_params=None):
        """Adds a sequence of vehicles to the list of vehicles in the network.

        Parameters
        ----------
        veh_id: str
            base vehicle ID for the vehicles (will be appended by a number)
        acceleration_controller: tup, optional
            1st element: flow-specified acceleration controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        lane_change_controller: tup, optional
            1st element: flow-specified lane-changer controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        routing_controller: tup, optional
            1st element: flow-specified routing controller
            2nd element: controller parameters (may be set to None to maintain
            default parameters)
        initial_speed: float, optional
            initial speed of the vehicles being added (in m/s)
        num_vehicles: int, optional
            number of vehicles of this type to be added to the network
        speed_mode: str or int, optional
            may be one of the following:

             * "no_collide" (default): Human and RL cars are preventing from
               reaching speeds that may cause crashes (also serves as a
               failsafe).
             * "aggressive": Human and RL cars are not limited by sumo with
               regard to their accelerations, and can crash longitudinally
             * "all_checks": all sumo safety checks are activated
             * "right_of_way": respect safe speed, right of way and
               brake hard at red lights if needed. DOES NOT respect
               max accel and decel which enables emergency stopping.
               Necessary to prevent custom models from crashing
             * int values may be used to define custom speed mode for the given
               vehicles, specified at:
               http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#speed_mode_.280xb3.29

        lane_change_mode: str or int, optional
            may be one of the following:

            * "strategic": Human cars make lane changes in accordance with SUMO
              to provide speed boosts
            * "no_lat_collide": Human cars will not make lane changes, RL cars
              can lane change into any space, no matter how likely it is to
              crash (default)
            * "aggressive": RL cars are not limited by sumo with regard to
              their lane-change actions, and can crash longitudinally
            * int values may be used to define custom lane change modes for the
              given vehicles, specified at:
              http://sumo.dlr.de/wiki/TraCI/Change_Vehicle_State#lane_change_mode_.280xb6.29

        sumo_car_following_params: flow.core.params.SumoCarFollowingParams type
            Params object specifying attributes for Sumo car following model.
        sumo_lc_params: flow.core.params.SumoLaneChangeParams type
            Params object specifying attributes for Sumo lane changing model.
        """
        if sumo_car_following_params is None:
            sumo_car_following_params = SumoCarFollowingParams()

        if sumo_lc_params is None:
            sumo_lc_params = SumoLaneChangeParams()

        type_params = {}
        type_params.update(sumo_car_following_params.controller_params)
        type_params.update(sumo_lc_params.controller_params)

        # If a vehicle is not sumo or RL, let the minGap be zero so that it
        # does not tamper with the dynamics of the controller
        if acceleration_controller[0] != SumoCarFollowingController \
                and acceleration_controller[0] != RLController:
            type_params["minGap"] = 0.0

        # adjust the speed mode value
        if isinstance(speed_mode, str) and speed_mode in SPEED_MODES:
            speed_mode = SPEED_MODES[speed_mode]
        elif not (isinstance(speed_mode, int)
                  or isinstance(speed_mode, float)):
            logging.error("Setting speed mode of {0} to "
                          "default.".format(veh_id))
            speed_mode = SPEED_MODES["no_collide"]

        # adjust the lane change mode value
        if isinstance(lane_change_mode, str) and lane_change_mode in LC_MODES:
            lane_change_mode = LC_MODES[lane_change_mode]
        elif not (isinstance(lane_change_mode, int)
                  or isinstance(lane_change_mode, float)):
            logging.error("Setting lane change mode of {0} to "
                          "default.".format(veh_id))
            lane_change_mode = LC_MODES["no_lat_collide"]

        # this dict will be used when trying to introduce new vehicles into
        # the network via a flow
        self.type_parameters[veh_id] = \
            {"acceleration_controller": acceleration_controller,
             "lane_change_controller": lane_change_controller,
             "routing_controller": routing_controller,
             "initial_speed": initial_speed,
             "speed_mode": speed_mode,
             "lane_change_mode": lane_change_mode,
             "sumo_car_following_params": sumo_car_following_params,
             "sumo_lc_params": sumo_lc_params}

        self.initial.append({
            "veh_id": veh_id,
            "acceleration_controller": acceleration_controller,
            "lane_change_controller": lane_change_controller,
            "routing_controller": routing_controller,
            "initial_speed": initial_speed,
            "num_vehicles": num_vehicles,
            "speed_mode": speed_mode,
            "lane_change_mode": lane_change_mode,
            "sumo_car_following_params": sumo_car_following_params,
            "sumo_lc_params": sumo_lc_params})

        # this is used to return the actual headways from the vehicles class
        self.minGap[veh_id] = type_params["minGap"]

        for i in range(num_vehicles):
            v_id = veh_id + '_%d' % i

            # add the vehicle to the list of vehicle ids
            self.__ids.append(v_id)

            self.__vehicles[v_id] = dict()

            # specify the type
            self.__vehicles[v_id]["type"] = veh_id

            # specify the acceleration controller class
            self.__vehicles[v_id]["acc_controller"] = \
                acceleration_controller[0](
                    v_id,
                    sumo_cf_params=sumo_car_following_params,
                    **acceleration_controller[1])

            # specify the lane-changing controller class
            self.__vehicles[v_id]["lane_changer"] = \
                lane_change_controller[0](veh_id=v_id,
                                          **lane_change_controller[1])

            # specify the routing controller class
            if routing_controller is not None:
                self.__vehicles[v_id]["router"] = \
                    routing_controller[0](veh_id=v_id,
                                          router_params=routing_controller[1])
            else:
                self.__vehicles[v_id]["router"] = None

            # specify the speed of vehicles at the start of a rollout
            self.__vehicles[v_id]["initial_speed"] = initial_speed

            # check if the vehicle is human-driven or autonomous
            if acceleration_controller[0] == RLController:
                self.__rl_ids.append(v_id)
            else:
                self.__human_ids.append(v_id)

                # check if the vehicle's lane-changing / acceleration actions
                # are controlled by sumo or not.
                if acceleration_controller[0] != SumoCarFollowingController:
                    self.__controlled_ids.append(v_id)
                if lane_change_controller[0] != SumoLaneChangeController:
                    self.__controlled_lc_ids.append(v_id)

            # specify the speed and lane change mode for the vehicle
            self.__vehicles[v_id]["speed_mode"] = speed_mode
            self.__vehicles[v_id]["lane_change_mode"] = lane_change_mode

        # update the variables for the number of vehicles in the network
        self.num_vehicles = len(self.__ids)
        self.num_rl_vehicles = len(self.__rl_ids)

        # increase the number of unique types of vehicles in the network, and
        # add the type to the list of types
        self.num_types += 1
        self.types.append({"veh_id": veh_id, "type_params": type_params})

    def update(self, vehicle_obs, sim_obs, env):
        """Updates the vehicle class with data pertaining to the vehicles at
        the current time step.

        The following actions are performed:

        * The state of all vehicles is modified to match their state at the
          current time step. This includes states specified by sumo, and states
          explicitly defined by flow, e.g. "absolute_position".
        * If vehicles exit the network, they are removed from the vehicles
          class, and newly departed vehicles are introduced to the class.

        Parameters
        ----------
        vehicle_obs: dict
            vehicle observations provided from sumo via subscriptions
        sim_obs: dict
            simulation observations provided from sumo via subscriptions
        env: Environment type
            state of the environment at the current time step
        """

        # remove exiting vehicles from the vehicles class
        for veh_id in sim_obs[tc.VAR_ARRIVED_VEHICLES_IDS]:
            if veh_id not in sim_obs[tc.VAR_TELEPORT_STARTING_VEHICLES_IDS]:
                self.remove(veh_id)
            else:
                # this is meant to resolve the KeyError bug when there are
                # collisions
                vehicle_obs[veh_id] = self.__sumo_obs[veh_id]

        # add entering vehicles into the vehicles class
        for veh_id in sim_obs[tc.VAR_DEPARTED_VEHICLES_IDS]:
            veh_type = env.traci_connection.vehicle.getTypeID(veh_id)
            if veh_id in self.get_ids():
                # this occurs when a vehicle is actively being removed and
                # placed again in the network to ensure a constant number of
                # total vehicles (e.g. GreenWaveEnv). In this case, the vehicle
                # is already in the class; its state data just needs to be
                # updated
                pass
            else:
                self._add_departed(veh_id, veh_type, env)

        if env.time_counter == 0:
            # reset all necessary values
            for veh_id in self.__rl_ids:
                self.set_state(veh_id, "last_lc", -float("inf"))
            self._num_departed.clear()
            self._num_arrived.clear()
            self.sim_step = env.sim_step
        else:
            # update the "last_lc" variable
            for veh_id in self.__rl_ids:
                prev_lane = self.get_lane(veh_id)
                if vehicle_obs[veh_id][tc.VAR_LANE_INDEX] != \
                        prev_lane and veh_id in self.__rl_ids:
                    self.set_state(veh_id, "last_lc", env.time_counter)

            # update the "absolute_position" variable
            for veh_id in self.__ids:
                prev_pos = env.get_x_by_id(veh_id)
                this_edge = vehicle_obs.get(veh_id, {}).get(tc.VAR_ROAD_ID, "")
                this_pos = vehicle_obs.get(veh_id, {}).get(
                    tc.VAR_LANEPOSITION, -1001)

                # in case the vehicle isn't in the network
                if this_edge == "":
                    self.set_absolute_position(veh_id, -1001)
                else:
                    change = env.scenario.get_x(this_edge, this_pos) - prev_pos
                    new_abs_pos = (self.get_absolute_position(veh_id) +
                                   change) % env.scenario.length
                    self.set_absolute_position(veh_id, new_abs_pos)

            # updated the list of departed and arrived vehicles
            self._num_departed.append(
                len(sim_obs[tc.VAR_DEPARTED_VEHICLES_IDS]))
            self._num_arrived.append(
                len(sim_obs[tc.VAR_ARRIVED_VEHICLES_IDS]))

        # update the "headway", "leader", and "follower" variables
        for veh_id in self.__ids:
            headway = vehicle_obs.get(veh_id, {}).get(tc.VAR_LEADER, None)
            # check for a collided vehicle or a vehicle with no leader
            if headway is None:
                self.__vehicles[veh_id]["leader"] = None
                self.__vehicles[veh_id]["follower"] = None
                self.__vehicles[veh_id]["headway"] = 1e+3
            else:
                vtype = self.get_state(veh_id, "type")
                min_gap = self.minGap[vtype]
                self.__vehicles[veh_id]["headway"] = headway[1] + min_gap
                self.__vehicles[veh_id]["leader"] = headway[0]
                try:
                    self.__vehicles[headway[0]]["follower"] = veh_id
                except KeyError:
                    pass

        # update the sumo observations variable
        self.__sumo_obs = vehicle_obs.copy()

        # update the lane leaders data for each vehicle
        self._multi_lane_headways(env)

        # make sure the rl vehicle list is still sorted
        self.__rl_ids.sort()

    def _add_departed(self, veh_id, veh_type, env):
        """Adds a vehicle that entered the network from an inflow or reset.

        Parameters
        ----------
        veh_id: str
            name of the vehicle
        veh_type: str
            type of vehicle, as specified to sumo
        env: Env type
            state of the environment at the current time step
        """
        if veh_type not in self.type_parameters:
            raise KeyError("Entering vehicle is not a valid type.")

        self.num_vehicles += 1
        self.__ids.append(veh_id)
        self.__vehicles[veh_id] = dict()

        # specify the type
        self.__vehicles[veh_id]["type"] = veh_type

        sumo_cf_params = \
            self.type_parameters[veh_type]["sumo_car_following_params"]

        # specify the acceleration controller class
        accel_controller = \
            self.type_parameters[veh_type]["acceleration_controller"]
        self.__vehicles[veh_id]["acc_controller"] = \
            accel_controller[0](veh_id,
                                sumo_cf_params=sumo_cf_params,
                                **accel_controller[1])

        # specify the lane-changing controller class
        lc_controller = \
            self.type_parameters[veh_type]["lane_change_controller"]
        self.__vehicles[veh_id]["lane_changer"] = \
            lc_controller[0](veh_id=veh_id, **lc_controller[1])

        # specify the routing controller class
        rt_controller = self.type_parameters[veh_type]["routing_controller"]
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
            if accel_controller[0] != SumoCarFollowingController:
                self.__controlled_ids.append(veh_id)
            if lc_controller[0] != SumoLaneChangeController:
                self.__controlled_lc_ids.append(veh_id)

        # subscribe the new vehicle
        env.traci_connection.vehicle.subscribe(
            veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
                     tc.VAR_ROAD_ID, tc.VAR_SPEED, tc.VAR_EDGES])
        env.traci_connection.vehicle.subscribeLeader(veh_id, 2000)

        # some constant vehicle parameters to the vehicles class
        self.set_length(
            veh_id, env.traci_connection.vehicle.getLength(veh_id))

        # set the absolute position of the vehicle
        self.set_absolute_position(veh_id, 0)

        # set the "last_lc" parameter of the vehicle
        self.set_state(veh_id, "last_lc", env.time_counter)

        # specify the initial speed
        self.__vehicles[veh_id]["initial_speed"] = \
            self.type_parameters[veh_type]["initial_speed"]

        # set the speed mode for the vehicle
        speed_mode = self.type_parameters[veh_type]["speed_mode"]
        self.__vehicles[veh_id]["speed_mode"] = speed_mode
        env.traci_connection.vehicle.setSpeedMode(veh_id, speed_mode)

        # set the lane changing mode for the vehicle
        lc_mode = self.type_parameters[veh_type]["lane_change_mode"]
        self.__vehicles[veh_id]["lane_change_mode"] = lc_mode
        env.traci_connection.vehicle.setLaneChangeMode(veh_id, lc_mode)

        # make sure that the order of rl_ids is kept sorted
        self.__rl_ids.sort()

    def remove(self, veh_id):
        """Removes a vehicle.

        Removes all traces of the vehicle from the vehicles class and all valid
        ID lists, and decrements the total number of vehicles in this class.

        Parameters
        ----------
        veh_id: str
            unique identifier of th vehicle to be removed
        """
        del self.__vehicles[veh_id]
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

    def test_set_speed(self, veh_id, speed):
        self.__sumo_obs[veh_id][tc.VAR_SPEED] = speed

    def set_absolute_position(self, veh_id, absolute_position):
        self.__vehicles[veh_id]["absolute_position"] = absolute_position

    def test_set_position(self, veh_id, position):
        self.__sumo_obs[veh_id][tc.VAR_LANEPOSITION] = position

    def test_set_edge(self, veh_id, edge):
        self.__sumo_obs[veh_id][tc.VAR_ROAD_ID] = edge

    def test_set_lane(self, veh_id, lane):
        self.__sumo_obs[veh_id][tc.VAR_LANE_INDEX] = lane

    def set_leader(self, veh_id, leader):
        self.__vehicles[veh_id]["leader"] = leader

    def set_follower(self, veh_id, follower):
        self.__vehicles[veh_id]["follower"] = follower

    def set_headway(self, veh_id, headway):
        self.__vehicles[veh_id]["headway"] = headway

    def get_ids(self):
        """Returns the names of all vehicles currently in the network."""
        return self.__ids

    def get_human_ids(self):
        """Returns the names of all non-rl vehicles currently in the
        network."""
        return self.__human_ids

    def get_controlled_ids(self):
        """Returns the names of all flow acceleration-controlled vehicles
        currently in the network."""
        return self.__controlled_ids

    def get_controlled_lc_ids(self):
        """Returns the names of all flow lane change-controlled vehicles
        currently in the network."""
        return self.__controlled_lc_ids

    def get_rl_ids(self):
        """Returns the names of all rl-controlled vehicles in the network."""
        return self.__rl_ids

    def set_observed(self, veh_id):
        """Adds a vehicle to the list of observed vehicles."""
        if veh_id not in self.__observed_ids:
            self.__observed_ids.append(veh_id)

    def remove_observed(self, veh_id):
        """Removes a vehicle from the list of observed vehicles."""
        if veh_id in self.__observed_ids:
            self.__observed_ids.remove(veh_id)

    def get_observed_ids(self):
        """Returns the list of observed vehicles."""
        return self.__observed_ids

    def get_ids_by_edge(self, edges):
        """Returns the names of all vehicles in the specified edge. If no
        vehicles are currently in the edge, then returns an empty list."""
        if isinstance(edges, (list, np.ndarray)):
            return sum([self.get_ids_by_edge(edge) for edge in edges], [])
        return self._ids_by_edge.get(edges, []) or []

    def get_inflow_rate(self, time_span):
        """Returns the inflow rate (in veh/hr) of vehicles from the network for
        the last **time_span** seconds."""
        if len(self._num_departed) == 0:
            return 0
        num_inflow = self._num_departed[-int(time_span / self.sim_step):]
        return 3600 * sum(num_inflow) / (len(num_inflow) * self.sim_step)

    def get_outflow_rate(self, time_span):
        """Returns the outflow rate (in veh/hr) of vehicles from the network
        for the last **time_span** seconds."""
        if len(self._num_arrived) == 0:
            return 0
        num_outflow = self._num_arrived[-int(time_span / self.sim_step):]
        return 3600 * sum(num_outflow) / (len(num_outflow) * self.sim_step)

    def get_num_arrived(self):
        """Returns the number of vehicles that arrived in the last
        time step"""
        if len(self._num_arrived) > 0:
            return self._num_arrived[-1]
        else:
            return 0

    def get_initial_speed(self, veh_id, error=-1001):
        """Returns the initial speed upon reset of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_initial_speed(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("initial_speed", error)

    def get_lane_change_mode(self, veh_id, error=-1001):
        """Returns the lane change mode value of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        int

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_change_mode(vehID, error)
                    for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_change_mode", error)

    def get_speed_mode(self, veh_id, error=-1001):
        """Returns the speed mode value of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        int

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_speed_mode(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("speed_mode", error)

    def get_speed(self, veh_id, error=-1001):
        """Returns the speed of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_speed(vehID, error) for vehID in veh_id]
        return self.__sumo_obs.get(veh_id, {}).get(tc.VAR_SPEED, error)

    def get_absolute_position(self, veh_id, error=-1001):
        """Returns the absolute position of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_absolute_position(vehID, error)
                    for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("absolute_position", error)

    def get_position(self, veh_id, error=-1001):
        """Returns the position of the vehicle relative to its current edge.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_position(vehID, error) for vehID in veh_id]
        return self.__sumo_obs.get(veh_id, {}).get(tc.VAR_LANEPOSITION, error)

    def get_edge(self, veh_id, error=""):
        """Returns the position of the specified vehicle (relative to the
        current edge).

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_edge(vehID, error) for vehID in veh_id]
        return self.__sumo_obs.get(veh_id, {}).get(tc.VAR_ROAD_ID, error)

    def get_lane(self, veh_id, error=-1001):
        """Returns the lane index of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        int

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane(vehID, error) for vehID in veh_id]
        return self.__sumo_obs.get(veh_id, {}).get(tc.VAR_LANE_INDEX, error)

    def set_length(self, veh_id, length):
        self.__vehicles[veh_id]["length"] = length

    def get_length(self, veh_id, error=-1001):
        """Returns the length of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_length(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("length", error)

    def get_acc_controller(self, veh_id, error=None):
        """Returns the acceleration controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_acc_controller(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("acc_controller", error)

    def get_lane_changing_controller(self, veh_id, error=None):
        """Returns the lane changing controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_changing_controller(vehID, error)
                    for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_changer", error)

    def get_routing_controller(self, veh_id, error=None):
        """Returns the routing controller of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        object

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_routing_controller(vehID, error)
                    for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("router", error)

    def get_route(self, veh_id, error=list()):
        """Returns the route of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list<str>

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_route(vehID, error) for vehID in veh_id]
        return self.__sumo_obs.get(veh_id, {}).get(tc.VAR_EDGES, error)

    def get_leader(self, veh_id, error=""):
        """Returns the leader of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_leader(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("leader", error)

    def get_follower(self, veh_id, error=""):
        """Returns the follower of the specified vehicle.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_follower(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("follower", error)

    def get_headway(self, veh_id, error=-1001):
        """Returns the headway of the specified vehicle(s).

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        float

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_headway(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("headway", error)

    def set_lane_headways(self, veh_id, lane_headways):
        self.__vehicles[veh_id]["lane_headways"] = lane_headways

    def get_lane_headways(self, veh_id, error=list()):
        """Returns the headways between the specified vehicle and the vehicle
        ahead of it in all lanes.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list<float>

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_headways(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_headways", error)

    def set_lane_leaders(self, veh_id, lane_leaders):
        self.__vehicles[veh_id]["lane_leaders"] = lane_leaders

    def get_lane_leaders(self, veh_id, error=list()):
        """Returns the leaders for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list<float>

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_leaders(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_leaders", error)

    def set_lane_tailways(self, veh_id, lane_tailways):
        self.__vehicles[veh_id]["lane_tailways"] = lane_tailways

    def get_lane_tailways(self, veh_id, error=list()):
        """Returns the tailways between the specified vehicle and the vehicle
        behind it in all lanes.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list<float>

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_tailways(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_tailways", error)

    def set_lane_followers(self, veh_id, lane_followers):
        self.__vehicles[veh_id]["lane_followers"] = lane_followers

    def get_lane_followers(self, veh_id, error=list()):
        """Returns the followers for the specified vehicle in all lanes.

        Parameters
        ----------
        veh_id : str or list<str>
            vehicle id, or list of vehicle ids
        error : list, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        list<str>

        """
        if isinstance(veh_id, (list, np.ndarray)):
            return [self.get_lane_followers(vehID, error) for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get("lane_followers", error)

    # TODO(ak): setting sumo observations?
    def set_state(self, veh_id, state_name, state):
        """Generic set function.

        Updates the state *state_name* of the vehicle with id *veh_id* with the
        value *state*.
        """
        self.__vehicles[veh_id][state_name] = state

    # TODO(ak): getting sumo observations?
    def get_state(self, veh_id, state_name, error=None):
        """Generic get function. Returns the value of *state_name* of the
        specified vehicles at the current time step.
        """
        if isinstance(veh_id, list):
            return [self.get_state(vehID, state_name, error)
                    for vehID in veh_id]
        return self.__vehicles.get(veh_id, {}).get(state_name, error)

    def _multi_lane_headways(self, env):
        """Computes the lane leaders/followers/headways/tailways for all
        vehicles in the network."""
        edge_list = env.scenario.get_edge_list()
        junction_list = env.scenario.get_junction_list()
        tot_list = edge_list + junction_list
        num_edges = (len(env.scenario.get_edge_list()) +
                     len(env.scenario.get_junction_list()))

        # maximum number of lanes in the network
        max_lanes = max([env.scenario.num_lanes(edge_id)
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
                                                   num_edges, env)

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

    def _multi_lane_headways_util(self, veh_id, edge_dict, num_edges, env):
        """Utility function for _multi_lane_headways()

        Computes the lane headways, tailways, leaders, and followers for the
        specified vehicle.

        Parameters
        ----------
        veh_id : str
            name of the vehicle
        edge_dict : dict < list<tuple> >
            Key = Edge name
                Index = lane index
                Element = sorted list of (vehicle id, position)

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
        num_lanes = env.scenario.num_lanes(this_edge)

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
                if index < len(positions) - 1:
                    # check if the index does not correspond to the current
                    # vehicle
                    if ids[index] == veh_id:
                        leader[lane] = ids[index + 1]
                        headway[lane] = (positions[index + 1] - this_pos -
                                         self.get_length(leader[lane]))
                    else:
                        leader[lane] = ids[index]
                        headway[lane] = positions[index] - this_pos \
                            - self.get_length(leader[lane])

                # you are in the back of the queue, the lane follower is in the
                # edges behind you
                if index > 0:
                    follower[lane] = ids[index - 1]
                    tailway[lane] = this_pos - positions[index - 1] \
                        - self.get_length(veh_id)

            # if lane leader not found, check next edges
            if leader[lane] == "":
                headway[lane], leader[lane] = self._next_edge_leaders(
                    veh_id, edge_dict, lane, num_edges, env)

            # if lane follower not found, check previous edges
            if follower[lane] == "":
                tailway[lane], follower[lane] = self._prev_edge_followers(
                    veh_id, edge_dict, lane, num_edges, env)

        return headway, tailway, leader, follower

    def _next_edge_leaders(self, veh_id, edge_dict, lane, num_edges, env):
        """Looks to the edges/junctions in front of the vehicle's current edge
        for potential leaders. This is currently done by only looking one
        edge/junction backwards.

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
            if len(env.scenario.next_edge(edge, lane)) == 0:
                break

            add_length += env.scenario.edge_length(edge)
            edge, lane = env.scenario.next_edge(edge, lane)[0]

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

    def _prev_edge_followers(self, veh_id, edge_dict, lane, num_edges, env):
        """Looks to the edges/junctions behind the vehicle's current edge for
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
            if len(env.scenario.prev_edge(edge, lane)) == 0:
                break

            edge, lane = env.scenario.prev_edge(edge, lane)[0]
            add_length += env.scenario.edge_length(edge)

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
