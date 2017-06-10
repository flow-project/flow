import logging
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

import traci
import sumolib
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step

from cistar.core.util import ensure_dir
import pdb
import collections

"""
This file provides the interface for controlling a SUMO simulation. Using the environment class, you can
start sumo, provide a scenario to specify a configuration and controllers, perform simulation steps, and
reset the simulation to an initial configuration.

SumoEnv must be be Serializable to allow for pickling of the policy.

This class cannot be used as is: you must extend it to implement an action applicator method, and
properties to define the MDP if you choose to use it with RLLab.

A reinforcement learning environment can be built using SumoEnvironment as a parent class by
adding the following functions:
 - action_space(self): specifies the action space of the rl vehicles
 - observation_space(self): specifies the observation space of the rl vehicles
 - apply_rl_action(self, rl_actions): Specifies the actions to be performed by rl_vehicles
 - getState(self):
 - compute_reward():

"""

COLORS = [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (255, 255, 0, 0), (0, 255, 255, 0), (255, 0, 255, 0),
          (255, 255, 255, 0)]


class SumoEnvironment(Env, Serializable):
    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        """ Base environment for all Sumo-based operations
        
        [description]

        Arguments:
            env_params {dictionary} -- [description]
            sumo_binary {string} -- Either "sumo" or "sumo-gui"
            sumo_params {dictionary} -- {"port": {integer} connection to SUMO,
                            "timestep": {float} default=0.01s, SUMO default=1.0s}
            scenario {Scenario} -- @see Scenario; abstraction of the SUMO XML files
                                    which specify the vehicles placed on the net
        
        Raises:
            ValueError -- Raised if a SUMO port not provided
        """
        Serializable.quick_init(self, locals())

        self.env_params = env_params
        self.sumo_binary = sumo_binary
        self.scenario = scenario
        self.sumo_params = sumo_params
        # timer: Represents number of steps taken
        self.timer = 0
        # vehicles: Key = Vehicle ID, Value = Dictionary describing the vehicle
        self.vehicles = collections.OrderedDict()
        # initial_state: Key = Vehicle ID, Entry = (type_id, route_id, lane_index, lane_pos, speed, pos)
        # Ordered dictionary used to keep neural net inputs in order
        self.initial_state = {}
        self.ids = []
        self.controlled_ids, self.rl_ids = [], []
        self.state = None
        self.obs_var_labels = []

        self.intersection_edges = []
        if hasattr(self.scenario, "intersection_edgestarts"):
            for intersection_tuple in self.scenario.intersection_edgestarts:
                self.intersection_edges.append(intersection_tuple[0])

        # SUMO Params
        if "port" not in sumo_params:
            self.port = sumolib.miscutils.getFreeSocketPort()
        else:
            self.port = sumo_params['port']

        if "time_step" in sumo_params:
            self.time_step = sumo_params["time_step"]
        else:
            self.time_step = 0.01  # 0.01 = default time step for our research

        if "emission_path" in sumo_params:
            data_folder = sumo_params['emission_path']
            ensure_dir(data_folder)
            self.emission_out = data_folder + "{0}-emission.xml".format(self.scenario.name)
        else:
            self.emission_out = None

        # Env Params
        if 'fail-safe' in env_params:
            self.fail_safe = env_params['fail-safe']
        else:
            self.fail_safe = 'instantaneous'

        if 'intersection_fail-safe' in env_params:
            if env_params["intersection_fail-safe"] not in ["left-right", "top-bottom", "None"]:
                raise ValueError('Intersection fail-safe must either be "left-right", "top-bottom", or "None"')
            self.intersection_fail_safe = env_params["intersection_fail-safe"]
        else:
            self.intersection_fail_safe = "None"

        # lane changing duration is always present in the environment,
        # but only used by sub-classes that apply lane changing
        if "lane_change_duration" in self.env_params:
            self.lane_change_duration = self.env_params['lane_change_duration'] / self.time_step
        else:
            self.lane_change_duration = 5 / self.time_step

        self.start_sumo()
        self.setup_initial_state()

    # TODO: never used...
    def restart_sumo(self, sumo_params, sumo_binary=None):
        self.traci_connection.close(False)
        if sumo_binary:
            self.sumo_binary = sumo_binary
        if "port" in sumo_params:
            self.port = sumo_params['port']

        if "emission_path" in sumo_params:
            data_folder = sumo_params['emission_path']
            ensure_dir(data_folder)
            self.emission_out = data_folder + "{0}-emission.xml".format(self.scenario.name)

        self.start_sumo()
        self.setup_initial_state()

    def start_sumo(self):
        logging.info(" Starting SUMO on port " + str(self.port))
        logging.debug(" Cfg file " + str(self.scenario.cfg))
        logging.debug(" Emission file: " + str(self.emission_out))
        logging.debug(" Time step: " + str(self.time_step))

        # TODO: find a better way to do this
        # Opening the I/O thread to SUMO
        cfg_file = self.scenario.cfg
        if "mode" in self.env_params and self.env_params["mode"] == "ec2":
            cfg_file = "/root/code/rllab/" + cfg_file

        sumo_call = [self.sumo_binary,
                     "-c", cfg_file,
                     "--remote-port", str(self.port),
                     "--step-length", str(self.time_step)]
        print("Traci on port: ", self.port)
        if self.emission_out:
            sumo_call.append("--emission-output")
            sumo_call.append(self.emission_out)

        subprocess.Popen(sumo_call, stdout=sys.stdout, stderr=sys.stderr)
        # self.proc = subprocess.Popen(sumo_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # sys.stderr

        logging.debug(" Initializing TraCI on port " + str(self.port) + "!")

        self.traci_connection = traci.connect(self.port)

        self.traci_connection.simulationStep()

        # Density = num vehicles / length (in meters)
        # so density in vehicles/km would be 1000 * self.density
        self.density = self.scenario.num_vehicles / self.scenario.net_params['length']

    def setup_initial_state(self):
        """
        Store initial state so that simulation can be reset at the end.
        TODO: Make traci calls as bulk as possible
        Initial state is a dictionary: key = vehicle IDs, value = state describing car
        """
        self.ids = self.traci_connection.vehicle.getIDList()
        self.controlled_ids.clear()
        self.rl_ids.clear()
        self.vehicles.clear()
        for i in self.ids:
            if self.traci_connection.vehicle.getTypeID(i) == "rl":
                self.rl_ids.append(i)
            else:
                self.controlled_ids.append(i)

        # create the list of colors used to different between different types of
        # vehicles visually on sumo's gui
        colors = {}
        key_index = 0
        color_choice = np.random.choice(len(COLORS))
        for key in self.scenario.type_params.keys():
            colors[key] = COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        for veh_id in self.ids:
            # import initial state from traci and place in vehicle dict
            vehicle = dict()
            vehicle["id"] = veh_id
            veh_type = self.traci_connection.vehicle.getTypeID(veh_id)
            vehicle["type"] = veh_type
            self.traci_connection.vehicle.setColor(veh_id, colors[veh_type])
            vehicle["edge"] = self.traci_connection.vehicle.getRoadID(veh_id)
            vehicle["position"] = self.traci_connection.vehicle.getLanePosition(veh_id)
            vehicle["lane"] = self.traci_connection.vehicle.getLaneIndex(veh_id)
            vehicle["speed"] = self.traci_connection.vehicle.getSpeed(veh_id)
            vehicle["length"] = self.traci_connection.vehicle.getLength(veh_id)
            vehicle["max_speed"] = self.traci_connection.vehicle.getMaxSpeed(veh_id)
            # vehicle["distance"] = self.traci_connection.vehicle.getDistance(veh_id)

            # specify acceleration controller
            controller_params = self.scenario.type_params[veh_type][1]
            if isinstance(controller_params[0], str):
                vehicle['controller'] = None
            else:
                vehicle['controller'] = controller_params[0](veh_id=veh_id, **controller_params[1])

            # specify lane-changing controller
            lane_changer_params = self.scenario.type_params[veh_type][2]
            if lane_changer_params is not None:
                vehicle['lane_changer'] = lane_changer_params[0](veh_id=veh_id, **lane_changer_params[1])
            else:
                vehicle['lane_changer'] = None

            self.vehicles[veh_id] = vehicle
            self.vehicles[veh_id]["absolute_position"] = self.get_x_by_id(veh_id)
            # the time step of the last lane change is always present in the environment,
            # but only used by sub-classes that apply lane changing
            if veh_id in self.rl_ids:
                self.vehicles[veh_id]['last_lc'] = -1 * self.lane_change_duration

            # set speed mode
            self.set_speed_mode(veh_id)

            # set lane change mode
            self.set_lane_change_mode(veh_id)

            # Saving initial state
            # route_id = self.traci_connection.vehicle.getRouteID(veh_id)
            route_id = "route" + vehicle["edge"]
            pos = self.traci_connection.vehicle.getPosition(veh_id)

            self.initial_state[veh_id] = (vehicle["type"], route_id, vehicle["lane"],
                                          vehicle["position"], vehicle["speed"], pos)

        # remove sumo controlled vehicles from controlled_ids
        self.controlled_ids = [self.controlled_ids[i] for i in range(len(self.controlled_ids)) if
                               self.vehicles[self.controlled_ids[i]]["controller"] is not None]
        print(self.controlled_ids)

        # dictionary of initial observations used while resetting vehicles after each rollout
        self.initial_observations = deepcopy(self.vehicles)

    def step(self, rl_actions):
        """
        Run one timestep of the environment's dynamics. "Self-driving cars" will
        step forward based on rl_actions, provided by the RL algorithm. Other cars
        will step forward based on their car following model. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        rl_actions : an action provided by the rl algorithm
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        try:
            self.timer += 1
            accel = []
            if self.sumo_params["traci_control"]:
                for veh_id in self.controlled_ids:
                    # acceleration action
                    action = self.vehicles[veh_id]['controller'].get_action(self)
                    accel.append(action)

                    # lane changing action
                    if self.scenario.lanes > 1: 
                        if self.vehicles[veh_id]['lane_changer'] is not None:
                            new_lane = self.vehicles[veh_id]['lane_changer'].get_action(self)
                            self.apply_lane_change([veh_id], target_lane=new_lane)

                self.apply_acceleration(self.controlled_ids, acc=accel)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            self.traci_connection.simulationStep()

            # a = self.proc.stderr.read()
            # print(a)

            for veh_id in self.ids:
                prev_pos = self.get_x_by_id(veh_id)
                self.vehicles[veh_id]["type"] = self.traci_connection.vehicle.getTypeID(veh_id)
                this_edge = self.traci_connection.vehicle.getRoadID(veh_id)
                if this_edge is None:
                    print('Null edge for vehicle:', veh_id)
                else:
                    self.vehicles[veh_id]["edge"] = this_edge
                self.vehicles[veh_id]["position"] = self.traci_connection.vehicle.getLanePosition(veh_id)
                self.vehicles[veh_id]["lane"] = self.traci_connection.vehicle.getLaneIndex(veh_id)
                veh_speed = self.traci_connection.vehicle.getSpeed(veh_id)
                self.vehicles[veh_id]["speed"] = veh_speed
                # self.vehicles[veh_id]["fuel"] = self.traci_connection.vehicle.getFuelConsumption(veh_id)
                # self.vehicles[veh_id]["distance"] = self.traci_connection.vehicle.getDistance(veh_id)
                try:
                    self.vehicles[veh_id]["absolute_position"] += \
                        (self.get_x_by_id(veh_id) - prev_pos) % self.scenario.length
                except ValueError:
                    self.vehicles[veh_id]["absolute_position"] = -1001

                if (self.traci_connection.vehicle.getDistance(veh_id) < 0 or
                            self.traci_connection.vehicle.getSpeed(veh_id) < 0):
                    print("Traci is returning error codes for some of your values", veh_id)

            # TODO: Can self._state be initialized, saved and updated so that we can exploit numpy speed
            # collect information of the state of the network based on the environment class used
            self.state = self.getState()

            # check whether any vehicles collided at any intersections
            intersection_crash = self.check_intersection_crash()

            # compute the reward
            reward = self.compute_reward(self.state, rl_actions, fail=intersection_crash)

            # TODO: Allow for partial observability
            next_observation = np.copy(self.state)

            if (self.traci_connection.simulation.getEndingTeleportNumber() != 0
                or self.traci_connection.simulation.getStartingTeleportNumber() != 0
                or any(self.state.flatten() == -1001)
                or intersection_crash):
                # Crash has occurred, end rollout
                if self.fail_safe == "None":
                    return Step(observation=next_observation, reward=reward, done=True)
                else:
                    print("Crash has occurred! Check failsafes!")
                    return Step(observation=next_observation, reward=reward, done=False)
            else:
                return Step(observation=next_observation, reward=reward, done=False)
        except KeyboardInterrupt:
            self.close() 

    # @property
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # create the list of colors used to visually distinguish between different types of vehicles
        colors = {}
        key_index = 0
        color_choice = np.random.choice(len(COLORS))
        for key in self.scenario.type_params.keys():
            colors[key] = COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        # re-initialize the perceived state
        self.vehicles = deepcopy(self.initial_observations)

        for veh_id in self.ids:
            type_id, route_id, lane_index, lane_pos, speed, pos = self.initial_state[veh_id]

            # clear controller acceleration queue of traci-controlled vehicles
            if not self.vehicles[veh_id]['type'] == 'rl' and self.vehicles[veh_id]['controller'] is not None:
                self.vehicles[veh_id]['controller'].reset_delay(self)

            # clear vehicles from traci connection and re-introduce vehicles with pre-defined initial position
            self.traci_connection.vehicle.remove(veh_id)
            self.traci_connection.vehicle.addFull(veh_id, route_id, typeID=str(type_id), departLane=str(lane_index),
                                                  departPos=str(lane_pos), departSpeed=str(speed))
            self.traci_connection.vehicle.setColor(veh_id, colors[self.vehicles[veh_id]['type']])

            # reset speed mode
            self.set_speed_mode(veh_id)

            # reset lane change mode
            self.set_lane_change_mode(veh_id)

        self.traci_connection.simulationStep()

        self.state = self.getState()
        observation = np.copy(self.state)

        return observation

    def additional_command(self):
        pass

    def apply_rl_actions(self, rl_actions):
        """
        Specifies the actions to be performed by rl_vehicles
        """
        raise NotImplementedError

    def apply_acceleration(self, veh_ids, acc):
        """
        Given an acceleration, set instantaneous velocity given that acceleration.
        Prevents vehicles from moves backwards (issuing negative velocities).

        :param veh_ids: vehicles to apply the acceleration to
        :param acc: requested accelerations from the vehicles
        :return acc_deviation: difference between the requested acceleration that keeps the velocity positive
        """
        # TODO: delete me? (start here)
        for i, veh_id in enumerate(veh_ids):
            # fail-safe to prevent longitudinal (bumper-to-bumper) crashing
            if self.fail_safe == 'instantaneous':
                safe_acc = self.vehicles[veh_id]['controller'].get_safe_action_instantaneous(self, acc[i])
            elif self.fail_safe == 'eugene':
                safe_acc = self.vehicles[veh_id]['controller'].get_safe_action(self, acc[i])
            else:
                safe_acc = acc[i]

            # fail-safe to prevent crashing at intersections
            if self.intersection_fail_safe != "None":
                safe_acc = self.vehicles[veh_id]['controller'].get_safe_intersection_action(self, safe_acc)

            acc[i] = safe_acc
        # TODO: delete me? (end here)

        # issue traci command for requested acceleration
        thisVel = np.array([self.vehicles[vid]['speed'] for vid in veh_ids])
        requested_nextVel = thisVel + np.array(acc) * self.time_step
        actual_nextVel = requested_nextVel.clip(min=0)

        for i, vid in enumerate(veh_ids):
            self.traci_connection.vehicle.slowDown(vid, actual_nextVel[i], 1)

        actual_acc = (actual_nextVel - thisVel) / self.time_step
        acc_deviation = (actual_nextVel - requested_nextVel) / self.time_step

        return actual_acc, acc_deviation

    def apply_lane_change(self, veh_ids, direction=None, target_lane=None):
        """
        Applies an instantaneous lane-change to a set of vehicles, while preventing vehicles from
        moving to lanes that do not exist.

        Takes as input either a set of directions or a target_lanes. If both are provided, a
        ValueError is raised.

        :param veh_ids: vehicles to apply the lane change to
        :param direction: array on intergers in {-1,1}; -1 to the right, 1 to the left
        :param target_lane: array of indeces of lane to enter
        :return: penalty value: 0 for successful lane change, -1 for impossible lane change
        """
        if direction is not None and target_lane is not None:
            raise ValueError("Cannot provide both a direction and target_lane.")
        elif direction is None and target_lane is None:
            raise ValueError("A direction or target_lane must be specified.")

        if self.scenario.lanes == 1:
            print("Uh oh, single lane track.")
            return -1

        current_lane = np.array([self.vehicles[vid]['lane'] for vid in veh_ids])

        if target_lane is None:
            target_lane = current_lane + direction

        safe_target_lane = np.clip(target_lane, 0, self.scenario.lanes - 1)

        lane_change_penalty = []
        for i, vid in enumerate(veh_ids):
            if safe_target_lane[i] == target_lane:
                self.traci_connection.vehicle.changeLane(vid, int(target_lane), 1)
                self.vehicles[vid]['last_lc'] = self.timer
                lane_change_penalty.append(0)
            else:
                lane_change_penalty.append(-1)

        return lane_change_penalty

    def set_speed_mode(self, veh_id):
        """

        :param veh_id:
        :return:
        """
        speed_mode = 1

        if "rl_sm" in self.sumo_params:
            if veh_id in self.rl_ids:
                if self.sumo_params["rl_sm"] == "aggressive":
                    speed_mode = 0
                elif self.sumo_params["rl_sm"] == "no_collide":
                    speed_mode = 1

        if "human_sm" in self.sumo_params:
            if veh_id not in self.rl_ids:
                if self.sumo_params["human_sm"] == "aggressive":
                    speed_mode = 0
                elif self.sumo_params["human_sm"] == "no_collide":
                    speed_mode = 1

        self.traci_connection.vehicle.setSpeedMode(veh_id, speed_mode)

    def set_lane_change_mode(self, veh_id):
        """

        :param veh_id:
        :return:
        """
        if self.scenario.lanes > 1:
            lc_mode = 768

            if "rl_lc" in self.sumo_params:
                if veh_id in self.rl_ids:
                    if self.sumo_params["rl_lc"] == "aggressive":
                        # Let TRACI make any lane changes it wants
                        lc_mode = 0
                    elif self.sumo_params["rl_lc"] == "no_lat_collide":
                        lc_mode = 256

            if "human_lc" in self.sumo_params:
                if veh_id not in self.rl_ids:
                    if self.sumo_params["human_lc"] == "strategic":
                        lc_mode = 853
                    else:
                        lc_mode = 768

            self.traci_connection.vehicle.setLaneChangeMode(veh_id, lc_mode)

    def check_intersection_crash(self):
        """
        Checks if two vehicles are moving through the same intersection from perpendicular ends

        :return: boolean value (True if crash occurred, False else)
        """
        if len(self.intersection_edges) == 0:
            return False
        else:
            return any([self.intersection_edges[0] in self.vehicles[veh_id]["edge"] for veh_id in self.ids]) \
                   and any([self.intersection_edges[1] in self.vehicles[veh_id]["edge"] for veh_id in self.ids])

    def check_longitudinal_crash(self):
        """
        Checks if the collision was the result of a forward movement (i.e. bumper-to-bumper crash)

        :return: boolean value (True if crash occurred, False else)
        """
        pass

    def check_lane_change_crash(self):
        """
        Checks if the collision was the result of a lane change

        :return: boolean value (True if crash occurred, False else)
        """
        pass

    def get_x_by_id(self, veh_id):
        """
        Returns the position of a vehicle relative to a certain reference (origin)
        """
        raise NotImplementedError

    def getState(self):
        """
        Returns the state of the simulation, dependent on the experiment/environment
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns an action space object
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns an observation space object
        """
        raise NotImplementedError

    def compute_reward(self, state, actions, **kwargs):
        """Reward function for RL.
        
        Arguments:
            state {Array-type} -- State of all the vehicles in the simulation
            fail {bool-type} -- represents any crash or fail not explicitly present in the state
        """
        raise NotImplementedError

    def render(self):
        """
        Description of the state for when verbose mode is on.
        """
        raise NotImplementedError

    def terminate(self):
        """
        Closes the TraCI I/O connection. Should be done at end of every experiment.
        Must be in Environment because the environment opens the TraCI connection.
        """
        self.traci_connection.close()

    def close(self):
        print("Aw damn, we closed")
        self.terminate()
