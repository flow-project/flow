import logging
import subprocess
import sys
from copy import deepcopy
import random

import numpy as np

import traci
from traci import constants as tc
import sumolib
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step

from cistar.controllers.base_controller import *
from cistar.controllers.car_following_models import *
from cistar.controllers.rlcontroller import RLController
from cistar.core.util import ensure_dir

import pdb
import collections
import time
import pickle

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
        # Ordered dictionary used to keep neural net inputs in order
        self.vehicles = collections.OrderedDict()
        # initial_state: Key = Vehicle ID, Entry = (type_id, route_id, lane_index, lane_pos, speed, pos)
        self.initial_state = {}
        self.ids = []
        self.controlled_ids, self.sumo_ids, self.rl_ids = [], [], []
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
            self.time_step = 0.1  # 0.1 = default time step for our research

        # parameter used to determine if initial conditions of vehicles are shuffled at reset
        if "vehicle_arrangement_shuffle" in sumo_params:
            self.vehicle_arrangement_shuffle = sumo_params["vehicle_arrangement_shuffle"]
        else:
            self.vehicle_arrangement_shuffle = False

        # parameter used to determine if starting position of vehicles should be updated between rollouts
        if "starting_position_shuffle" in sumo_params:
            self.starting_position_shuffle = sumo_params["starting_position_shuffle"]
        else:
            self.starting_position_shuffle = False

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
            self.fail_safe = 'None'

        # observation (sensor) noise associated with velocity data
        if "observation_vel_std" in env_params:
            self.observation_vel_std = env_params["observation_vel_std"]
        else:
            self.observation_vel_std = 0

        # observation (sensor) noise associated with position data
        if "observation_pos_std" in env_params:
            self.observation_pos_std = env_params["observation_pos_std"]
        else:
            self.observation_pos_std = 0

        # action (actuator) noise associated with human-driven vehicle acceleration
        if "human_acc_std" in env_params:
            self.human_acc_std = env_params["human_acc_std"]
        else:
            self.human_acc_std = 0

        # action (actuator) noise associated with autonomous vehicle acceleration
        if "rl_acc_std" in env_params:
            self.rl_acc_std = env_params["rl_acc_std"]
        else:
            self.rl_acc_std = 0

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

        self.traci_connection = traci.connect(self.port, numRetries=100)

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
        # self.pos = dict()
        # self.vel = dict()
        # self.lanes = dict()

        # collect ids and prepare id and vehicle lists
        self.ids = self.traci_connection.vehicle.getIDList()
        self.controlled_ids.clear()
        self.sumo_ids.clear()
        self.rl_ids.clear()
        self.vehicles.clear()

        # create the list of colors used to different between different types of
        # vehicles visually on sumo's gui
        self.colors = {}
        key_index = 1
        color_choice = np.random.choice(len(COLORS))
        for key in self.scenario.type_params.keys():
            self.colors[key] = COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        for veh_id in self.ids:
            # import initial state from traci and place in vehicle dict
            vehicle = dict()
            vehicle["id"] = veh_id
            veh_type = self.traci_connection.vehicle.getTypeID(veh_id)
            vehicle["type"] = veh_type
            self.traci_connection.vehicle.setColor(veh_id, self.colors[veh_type])
            vehicle["edge"] = self.traci_connection.vehicle.getRoadID(veh_id)
            vehicle["position"] = self.traci_connection.vehicle.getLanePosition(veh_id)
            vehicle["lane"] = self.traci_connection.vehicle.getLaneIndex(veh_id)
            vehicle["speed"] = self.traci_connection.vehicle.getSpeed(veh_id)
            vehicle["length"] = self.traci_connection.vehicle.getLength(veh_id)
            vehicle["max_speed"] = self.traci_connection.vehicle.getMaxSpeed(veh_id)

            # specify acceleration controller
            controller_params = self.scenario.type_params[veh_type][1]
            vehicle['controller'] = controller_params[0](veh_id=veh_id, **controller_params[1])

            if controller_params[0] == SumoController:
                self.sumo_ids.append(veh_id)
            elif controller_params[0] == RLController:
                self.rl_ids.append(veh_id)
            else:
                self.controlled_ids.append(veh_id)

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

            # self.pos[veh_id] = [self.vehicles[veh_id]["absolute_position"]]
            # self.vel[veh_id] = [self.vehicles[veh_id]["speed"]]
            # self.lanes[veh_id] = [self.vehicles[veh_id]["lane"]]

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect headway, leader id, and follower id data
        vehicles = self.get_headway_dict()

        for veh_id in self.ids:
            self.vehicles[veh_id]["headway"] = vehicles[veh_id]["headway"]
            self.vehicles[veh_id]["leader"] = vehicles[veh_id]["leader"]
            self.vehicles[veh_id]["follower"] = vehicles[veh_id]["follower"]

        # dictionary of initial observations used while resetting vehicles after each rollout
        self.initial_observations = deepcopy(dict(self.vehicles))

        # contains the last lc before the current step
        self.prev_last_lc = dict()
        for veh_id in self.ids:
            self.prev_last_lc[veh_id] = self.vehicles[veh_id]["last_lc"]

        for veh_id in self.ids:
            self.traci_connection.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
                                                             tc.VAR_ROAD_ID, tc.VAR_SPEED])
            self.traci_connection.vehicle.subscribeLeader(veh_id, 2000)

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
        self.timer += 1

        # perform acceleration and (optionally) lane change actions for cistar-controlled human-driven vehicles
        accel = []
        if len(self.controlled_ids) > 0:
            for veh_id in self.controlled_ids:
                # acceleration action
                action = self.vehicles[veh_id]['controller'].get_action(self)
                accel.append(action)

                # lane changing action
                if self.scenario.lanes > 1:
                    if self.vehicles[veh_id]['lane_changer'] is not None:
                        new_lane = self.vehicles[veh_id]['lane_changer'].get_action(self)
                        self.apply_lane_change([veh_id], target_lane=[new_lane])

            self.apply_acceleration(self.controlled_ids, acc=accel)

        # perform (optionally) lane change actions for sumo-controlled human-driven vehicles
        if len(self.sumo_ids) > 0:
            for veh_id in self.sumo_ids:
                # lane changing action
                if self.scenario.lanes > 1:
                    if self.vehicles[veh_id]['lane_changer'] is not None:
                        new_lane = self.vehicles[veh_id]['lane_changer'].get_action(self)
                        self.apply_lane_change([veh_id], target_lane=[new_lane])

        self.apply_rl_actions(rl_actions)

        self.additional_command()

        self.traci_connection.simulationStep()

        # store new observations in the network after traci simulation step
        network_observations = self.traci_connection.vehicle.getSubscriptionResults()

        crash = False
        for veh_id in self.ids:
            prev_pos = self.get_x_by_id(veh_id)
            prev_lane = self.vehicles[veh_id]["lane"]
            try:
                self.vehicles[veh_id]["position"] = network_observations[veh_id][tc.VAR_LANEPOSITION]
            except KeyError:
                del self.vehicles[veh_id]
                self.ids.remove(veh_id)
                if veh_id in self.rl_ids:
                    self.rl_ids.remove(veh_id)
                elif veh_id in self.controlled_ids:
                    self.controlled_ids.remove(veh_id)
                else:
                    self.sumo_ids.remove(veh_id)
                continue
            self.vehicles[veh_id]["edge"] = network_observations[veh_id][tc.VAR_ROAD_ID]
            self.vehicles[veh_id]["lane"] = network_observations[veh_id][tc.VAR_LANE_INDEX]
            if self.vehicles[veh_id]["lane"] != prev_lane and veh_id in self.rl_ids:
                self.vehicles[veh_id]["last_lc"] = self.timer
            self.vehicles[veh_id]["speed"] = network_observations[veh_id][tc.VAR_SPEED]

            try:
                change = self.get_x_by_id(veh_id) - prev_pos
                if change < 0:
                    change += self.scenario.length
                self.vehicles[veh_id]["absolute_position"] += change
            except ValueError or TypeError:
                self.vehicles[veh_id]["absolute_position"] = -1001

            if self.vehicles[veh_id]["position"] < 0 or self.vehicles[veh_id]["speed"] < 0:
                crash = True

            # Grab the headway
            headway = network_observations[veh_id][tc.VAR_LEADER]
            if headway is None:
                self.vehicles[veh_id]["leader"] = None
                self.vehicles[veh_id]["follower"] = None
                self.vehicles[veh_id]["headway"] = np.inf
            else:
                self.vehicles[veh_id]["headway"] = headway[1]
                self.vehicles[veh_id]["leader"] = headway[0]
                self.vehicles[headway[0]]["follower"] = veh_id

            # self.pos[veh_id].append(self.vehicles[veh_id]["absolute_position"])
            # self.vel[veh_id].append(self.vehicles[veh_id]["speed"])
            # self.lanes[veh_id].append(self.vehicles[veh_id]["lane"])

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect information of the state of the network based on the environment class used
        if self.scenario.num_rl_vehicles > 0: 
            self.state = self.getState()
        else:
            self.state = []
        # collect observation new state associated with action
        next_observation = np.copy(self.state)

        # crash encodes whether sumo experienced a crash
        crash = crash or self.traci_connection.simulation.getEndingTeleportNumber() != 0 \
            or self.traci_connection.simulation.getStartingTeleportNumber() != 0

        # compute the reward
        if self.scenario.num_rl_vehicles > 0:
            reward = self.compute_reward(self.state, rl_actions, fail=crash)
        else:
            reward = 0

        # if self.timer == 1500:
        #     output = {"pos": self.pos, "vel": self.vel, "lane": self.lanes}
        #
        #     with open("/home/aboudy/Documents/output.pkl", "wb") as handle:
        #         pickle.dump(output, handle)

        if crash:
            # Crash has occurred, end rollout
            if self.fail_safe == "None":
                return Step(observation=next_observation, reward=reward, done=True)
            else:
                print("Crash has occurred! Check failsafes!")
                return Step(observation=next_observation, reward=reward, done=True)
        else:
            return Step(observation=next_observation, reward=reward, done=False)

    # @property
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # create the list of colors used to visually distinguish between different types of vehicles
        self.timer = 0
        self.colors = {}
        key_index = 1
        color_choice = np.random.choice(len(COLORS))
        for key in self.scenario.type_params.keys():
            self.colors[key] = COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        # perform shuffling (if requested)
        if self.starting_position_shuffle or self.vehicle_arrangement_shuffle:
            if self.starting_position_shuffle:
                x0 = np.random.uniform(0, self.scenario.length)
            else:
                x0 = 1

            veh_ids = deepcopy([x[1] for x in self.scenario.generator.vehicle_ids])
            if self.vehicle_arrangement_shuffle:
                random.shuffle(veh_ids)

            initial_positions, initial_lanes = self.scenario.generate_starting_positions(x0=x0)

            initial_state = dict()
            for i, veh_id in enumerate(veh_ids):
                route_id = "route" + initial_positions[i][0]

                # replace initial routes, lanes, and positions to reflect new values
                list_initial_state = list(self.initial_state[veh_id])
                list_initial_state[1] = route_id
                list_initial_state[2] = initial_lanes[i]
                list_initial_state[3] = initial_positions[i][1]
                initial_state[veh_id] = tuple(list_initial_state)

                # replace initial positions in initial observations
                self.initial_observations[veh_id]["edge"] = initial_positions[i][0]
                self.initial_observations[veh_id]["position"] = initial_positions[i][1]

            self.initial_state = deepcopy(initial_state)

        # re-initialize the perceived state
        self.vehicles = deepcopy(self.initial_observations)

        # re-initialize memory on last lc
        self.prev_last_lc = dict()
        for veh_id in self.ids:
            self.vehicles[veh_id]["absolute_position"] = self.get_x_by_id(veh_id)
            self.prev_last_lc[veh_id] = self.vehicles[veh_id]["last_lc"]

        # reset the list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect headway, leader id, and follower id data
        vehicles = self.get_headway_dict()

        for veh_id in self.ids:
            self.vehicles[veh_id]["headway"] = vehicles[veh_id]["headway"]
            self.vehicles[veh_id]["leader"] = vehicles[veh_id]["leader"]
            self.vehicles[veh_id]["follower"] = vehicles[veh_id]["follower"]

            type_id, route_id, lane_index, lane_pos, speed, pos = self.initial_state[veh_id]

            # clear controller acceleration queue of traci-controlled vehicles
            if veh_id in self.controlled_ids:
                self.vehicles[veh_id]['controller'].reset_delay(self)

            # clear vehicles from traci connection and re-introduce vehicles with pre-defined initial position
            self.traci_connection.vehicle.remove(veh_id)
            self.traci_connection.vehicle.addFull(veh_id, route_id, typeID=str(type_id), departLane=str(lane_index),
                                                  departPos=str(lane_pos), departSpeed=str(speed))
            self.traci_connection.vehicle.setColor(veh_id, self.colors[self.vehicles[veh_id]['type']])

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
        pass

    def apply_acceleration(self, veh_ids, acc):
        """
        Given an acceleration, set instantaneous velocity given that acceleration.
        Prevents vehicles from moves backwards (issuing negative velocities).
        :param veh_ids: vehicles to apply the acceleration to
        :param acc: requested accelerations from the vehicles
        :return acc_deviation: difference between the requested acceleration that keeps the velocity positive
        """
        rl_i = 0
        human_i = 0
        for i, veh_id in enumerate(veh_ids):
            # add actuator noise to accelerations
            if veh_id in self.rl_ids:
                acc[i] += np.random.normal(0, self.rl_acc_std)
                rl_i += 1
            elif veh_id in self.controlled_ids:
                acc[i] += np.random.normal(0, self.human_acc_std)
                human_i += 1

            # fail-safe to prevent longitudinal (bumper-to-bumper) crashing
            if self.fail_safe == 'instantaneous':
                safe_acc = self.vehicles[veh_id]['controller'].get_safe_action_instantaneous(self, acc[i])
            elif self.fail_safe == 'eugene':
                safe_acc = self.vehicles[veh_id]['controller'].get_safe_action(self, acc[i])
            else:
                safe_acc = acc[i]

            acc[i] = safe_acc

        # issue traci command for requested acceleration
        thisVel = np.array([self.vehicles[vid]['speed'] for vid in veh_ids])
        requested_nextVel = thisVel + np.array(acc) * self.time_step
        actual_nextVel = requested_nextVel.clip(min=0)

        for i, vid in enumerate(veh_ids):
            self.traci_connection.vehicle.slowDown(vid, actual_nextVel[i], 1)

    def apply_lane_change(self, veh_ids, direction=None, target_lane=None):
        """
        Applies an instantaneous lane-change to a set of vehicles, while preventing vehicles from
        moving to lanes that do not exist.
        Takes as input either a set of directions or a target_lanes. If both are provided, a
        ValueError is raised.
        :param veh_ids: vehicles to apply the lane change to
        :param direction: array on integers in {-1,1}; -1 to the right, 1 to the left
        :param target_lane: array of indices of lane to enter
        :return: penalty value: 0 for successful lane change, -1 for impossible lane change
        """
        if direction is not None and target_lane is not None:
            raise ValueError("Cannot provide both a direction and target_lane.")
        elif direction is None and target_lane is None:
            raise ValueError("A direction or target_lane must be specified.")

        for veh_id in veh_ids:
            self.prev_last_lc[veh_id] = self.vehicles[veh_id]["last_lc"]

        if self.scenario.lanes == 1:
            print("Uh oh, single lane track.")
            return -1

        current_lane = np.array([self.vehicles[vid]['lane'] for vid in veh_ids])

        if target_lane is None:
            target_lane = current_lane + direction

        safe_target_lane = np.clip(target_lane, 0, self.scenario.lanes - 1)

        for i, vid in enumerate(veh_ids):
            if vid in self.rl_ids:
                if safe_target_lane[i] == target_lane[i] and target_lane[i] != current_lane[i]:
                        self.traci_connection.vehicle.changeLane(vid, int(target_lane[i]), 100000)
            else:
                self.traci_connection.vehicle.changeLane(vid, int(target_lane[i]), 100000)

    def choose_routes(self, veh_ids, route_choices):
        """
        Updates the route choice of vehicles in the network.
        :param veh_ids: list of vehicle identifiers
        :param route_choices: list of edges the vehicle wishes to traverse, starting with the edge the
               vehicle is currently on
        """
        pass

    def set_speed_mode(self, veh_id):
        # TODO: document
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
        Specifies the SUMO-defined lane-changing mode used to constrain lane-changing actions

        The available lane-changing modes are as follows:
         - default: Human and RL cars can only safely change into lanes
         - "strategic": Human cars make lane changes in accordance with SUMO to provide speed boosts
         - "no_lat_collide": RL cars can lane change into any space, no matter how likely it is to crash
         - "aggressive": RL cars can crash longitudinally
        """
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

    def check_longitudinal_crash(self):
        """
        Checks if the collision was the result of a forward movement (i.e. bumper-to-bumper crash)
        :return: boolean value (True if crash occurred, False else)
        """
        # TODO: figure out how to implement
        pass

    def check_lane_change_crash(self):
        """
        Checks if the collision was the result of a lane change
        :return: boolean value (True if crash occurred, False else)
        """
        # TODO: figure out how to implement
        pass

    def get_x_by_id(self, veh_id):
        """
        Returns the position of a vehicle relative to a certain reference (origin)
        """
        raise NotImplementedError

    def sort_by_position(self):
        """
        sorts the vehicle ids of vehicles in the network by position
        :return: a list of sorted vehicle ids
                 an extra component (list, tuple, etc...) containing extra sorted data, such as positions.
                  If no extra component is needed, a value of None should be returned
        """
        raise NotImplementedError

    # TODO: we are using traci calls for this now. Should we delete?
    def get_headway_dict(self):
        """
        Returns the headway between the current vehicle and the one in front of it
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

    def compute_reward(self, state, rl_actions, **kwargs):
        """Reward function for RL.
        
        Arguments:
            state {Array-type} -- State of all the vehicles in the simulation
            rl_actions {Array-type} -- array of actions performed by rl vehicles
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

        # output = {"pos": self.pos, "vel": self.vel}
        #
        # with open("/home/aboudy/Documents/output.pkl", "wb") as handle:
        #     pickle.dump(output, handle)

    def close(self):
        self.terminate()
