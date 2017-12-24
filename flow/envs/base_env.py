import logging
import subprocess
import sys
from copy import deepcopy
import time
import traceback

import traci
from traci import constants as tc
import gym

import sumolib

try:
    # Import serialiable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError as e:
    Serializable = object

import flow.core.config as config
from flow.controllers.car_following_models import *
from flow.core.util import ensure_dir

# Number of retries on restarting SUMO before giving up
RETRIES_ON_ERROR = 10

COLORS = [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (255, 255, 0, 0),
          (0, 255, 255, 0), (255, 0, 255, 0), (255, 255, 255, 0)]


class SumoEnvironment(gym.Env, Serializable):
    def __init__(self, env_params, sumo_params, scenario):
        """
        Base environment class. Provides the interface for controlling a SUMO
        simulation. Using this class, you can start sumo, provide a scenario to
        specify a configuration and controllers, perform simulation steps, and
        reset the simulation to an initial configuration.

        SumoEnvironment is Serializable to allow for pickling of the policy.

        This class cannot be used as is: you must extend it to implement an
        action applicator method, and properties to define the MDP if you choose
        to use it with RLLab. This can be done by overloading the following
        functions in a child class:
         - action_space
         - observation_space
         - apply_rl_action
         - get_state
         - compute_reward

         Attributes
         ----------
         env_params: EnvParams type:
            see flow/core/params.py
         sumo_params: SumoParams type
            see flow/core/params.py
        scenario: Scenario type
            see flow/scenarios/base_scenario.py
        """
        # Invoke serialiable if using rllab
        if Serializable is not object:
            Serializable.quick_init(self, locals())

        self.env_params = env_params
        self.scenario = scenario
        self.sumo_params = sumo_params
        self.sumo_binary = self.sumo_params.sumo_binary
        self.vehicles = scenario.vehicles
        # timer: Represents number of steps taken since the start of a rollout
        self.timer = 0
        # initial_state:
        #   Key = Vehicle ID,
        #   Entry = (type_id, route_id, lane_index, lane_pos, speed, pos)
        self.initial_state = {}
        # vehicle identifiers for all vehicles
        self.ids = []
        # vehicle identifiers for specific types of vehicles
        self.controlled_ids, self.sumo_ids, self.rl_ids = [], [], []
        self.state = None
        self.obs_var_labels = []

        # SUMO Params
        if sumo_params.port:
            self.port = sumo_params.port
        else:
            self.port = sumolib.miscutils.getFreeSocketPort()
        self.time_step = sumo_params.time_step
        self.vehicle_arrangement_shuffle = \
            sumo_params.vehicle_arrangement_shuffle
        self.starting_position_shuffle = sumo_params.starting_position_shuffle
        self.emission_path = sumo_params.emission_path

        # path to the output (emission) file provided by sumo
        if self.emission_path:
            ensure_dir(self.emission_path)
            self.emission_out = \
                self.emission_path + "{0}-emission.xml".format(
                    self.scenario.name)
        else:
            self.emission_out = None

        self.fail_safe = env_params.fail_safe
        self.max_speed = env_params.max_speed
        self.lane_change_duration = \
            env_params.get_lane_change_duration(self.time_step)
        self.shared_reward = env_params.shared_reward
        self.shared_policy = env_params.shared_policy

        # the available_routes variable contains a dictionary of routes vehicles
        # can traverse; to be used when routes need to be chosen dynamically
        self.available_routes = self.scenario.generator.rts

        # Check if we are in a multi-agent scenario
        if isinstance(self.action_space, list):
            self.multi_agent = True
        else:
            self.multi_agent = False

        self.start_sumo()
        self.setup_initial_state()

    def restart_sumo(self, sumo_params, sumo_binary=None):
        """
        Restarts an already initialized environment. Used when visualizing a
        rollout.
        """
        self.traci_connection.close(False)
        if sumo_binary:
            self.sumo_binary = sumo_binary

        self.port = sumolib.miscutils.getFreeSocketPort()
        if self.emission_path:
            data_folder = self.emission_path
            ensure_dir(data_folder)
            self.emission_out = \
                data_folder + "{0}-emission.xml".format(self.scenario.name)

        self.start_sumo()
        self.setup_initial_state()

    def start_sumo(self):
        """
        Starts a sumo instance using the configuration files created by the
        generator class. Also initializes a traci connection to interface with
        sumo from Python.
        """
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                logging.info(" Starting SUMO on port " + str(self.port))
                logging.debug(" Cfg file " + str(self.scenario.cfg))
                logging.debug(" Emission file: " + str(self.emission_out))
                logging.debug(" Time step: " + str(self.time_step))

                # Opening the I/O thread to SUMO
                cfg_file = self.scenario.cfg
                # if "mode" in self.env_params and self.env_params["mode"] == "ec2":
                #     cfg_file = "/root/code/rllab/" + cfg_file

                sumo_call = [self.sumo_binary,
                             "-c", cfg_file,
                             "--remote-port", str(self.port),
                             "--step-length", str(self.time_step)]
                logging.info("Traci on port: ", self.port)
                if self.emission_out:
                    sumo_call.append("--emission-output")
                    sumo_call.append(self.emission_out)

                subprocess.Popen(sumo_call, stdout=sys.stdout, stderr=sys.stderr)

                logging.debug(" Initializing TraCI on port " + str(self.port) + "!")

                # wait a small period of time for the subprocess to activate before
                # trying to connect with traci
                time.sleep(config.SUMO_SLEEP)

                self.traci_connection = traci.connect(self.port, numRetries=100)

                self.traci_connection.simulationStep()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                error = e
        raise error

    def setup_initial_state(self):
        """
        Returns information on the initial state of the vehicles in the network,
        to be used upon reset.
        Also adds initial state information to the self.vehicles class and
        starts a subscription with sumo to collect state information each step.

        Returns
        -------
        initial_observations: dictionary
            key = vehicles IDs
            value = state describing car at the start of the rollout
        initial_state: dictionary
            key = vehicles IDs
            value = sparse state information (only what is needed to add a
            vehicle in a sumo network with traci)
        """
        # collect the ids of vehicles in the network
        self.ids = self.vehicles.get_ids()
        self.controlled_ids = self.vehicles.get_controlled_ids()
        self.sumo_ids = self.vehicles.get_sumo_ids()
        self.rl_ids = self.vehicles.get_rl_ids()

        # dictionary of initial observations used while resetting vehicles after
        # each rollout
        self.initial_observations = dict.fromkeys(self.ids)

        # create the list of colors used to different between different types of
        # vehicles visually on sumo's gui
        self.colors = {}
        key_index = 1
        color_choice = np.random.choice(len(COLORS))
        for i in range(self.vehicles.num_types):
            self.colors[self.vehicles.types[i]] = \
                COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        for veh_id in self.ids:
            # set the colors of the vehicles based on their unique types
            veh_type = self.vehicles.get_state(veh_id, "type")
            self.traci_connection.vehicle.setColor(veh_id,
                                                   self.colors[veh_type])

            # add the initial states to the vehicles class
            self.vehicles.set_edge(
                veh_id, self.traci_connection.vehicle.getRoadID(veh_id))
            self.vehicles.set_position(
                veh_id, self.traci_connection.vehicle.getLanePosition(veh_id))
            self.vehicles.set_lane(
                veh_id, self.traci_connection.vehicle.getLaneIndex(veh_id))
            self.vehicles.set_speed(
                veh_id, self.traci_connection.vehicle.getSpeed(veh_id))
            self.vehicles.set_route(
                veh_id, self.available_routes[self.vehicles.get_edge(veh_id)])
            self.vehicles.set_absolute_position(
                veh_id, self.get_x_by_id(veh_id))
            # the time step of the last lane change is always present in
            # the environment,but only used by sub-classes that apply lane
            # changing
            self.vehicles.set_state(veh_id, "last_lc",
                                    -1 * self.lane_change_duration)
            # some constant vehicle parameters
            self.vehicles.set_state(
                veh_id, "length",
                self.traci_connection.vehicle.getLength(veh_id))
            self.vehicles.set_state(veh_id, "max_speed", self.max_speed)

            # import initial state data to initial_observations dict
            self.initial_observations[veh_id] = dict()
            self.initial_observations[veh_id]["type"] = veh_type
            self.initial_observations[veh_id]["edge"] = \
                self.traci_connection.vehicle.getRoadID(veh_id)
            self.initial_observations[veh_id]["position"] = \
                self.traci_connection.vehicle.getLanePosition(veh_id)
            self.initial_observations[veh_id]["lane"] = \
                self.traci_connection.vehicle.getLaneIndex(veh_id)
            self.initial_observations[veh_id]["speed"] = \
                self.traci_connection.vehicle.getSpeed(veh_id)
            self.initial_observations[veh_id]["route"] = \
                self.available_routes[self.initial_observations[veh_id]["edge"]]
            self.initial_observations[veh_id]["absolute_position"] = \
                self.get_x_by_id(veh_id)

            # set speed mode
            self.set_speed_mode(veh_id)

            # set lane change mode
            self.set_lane_change_mode(veh_id)

            # save the initial state. This is used in the _reset function
            #
            route_id = "route" + self.initial_observations[veh_id]["edge"]
            pos = self.traci_connection.vehicle.getPosition(veh_id)

            self.initial_state[veh_id] = \
                (self.initial_observations[veh_id]["type"], route_id,
                 self.initial_observations[veh_id]["lane"],
                 self.initial_observations[veh_id]["position"],
                 self.initial_observations[veh_id]["speed"], pos)

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect headway, leader id, and follower id data
        for veh_id in self.ids:
            headway = self.traci_connection.vehicle.getLeader(veh_id, 2000)
            if headway is None:
                self.vehicles.set_leader(veh_id, None)
                self.vehicles.set_headway(veh_id, 9e9)
            else:
                self.vehicles.set_leader(veh_id, headway[0])
                self.vehicles.set_headway(veh_id, headway[1])
                self.vehicles.set_follower(headway[0], veh_id)

        # contains the last lc before the current step
        self.prev_last_lc = dict()
        for veh_id in self.ids:
            self.prev_last_lc[veh_id] = self.vehicles.get_state(veh_id,
                                                                "last_lc")

        # subscribe the requested states for traci-related speedups
        for veh_id in self.ids:
            self.traci_connection.vehicle.subscribe(
                veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION,
                         tc.VAR_ROAD_ID, tc.VAR_SPEED])
            self.traci_connection.vehicle.subscribeLeader(veh_id, 2000)

    def _step(self, rl_actions):
        """
        Run one timestep of the environment's dynamics. An autonomous agent
        (i.e. autonomous vehicles) performs an action provided by the RL
        algorithm. Other cars step forward based on their car following model.
        When end of episode is reached, reset() should be called to reset the
        environment's initial state.

        Parameters
        ----------
        rl_actions: numpy ndarray
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation: numpy ndarray
            agent's observation of the current environment
        reward: float
            amount of reward associated with the previous state/action pair
        done: boolean
            indicates whether the episode has ended
        info: dictionary
            contains other diagnostic information from the previous action
        """
        self.timer += 1

        # perform acceleration and (optionally) lane change actions for
        # traci-controlled human-driven vehicles
        if len(self.controlled_ids) > 0:
            accel = []
            for veh_id in self.controlled_ids:
                # acceleration action
                action = self.vehicles.get_acc_controller(veh_id).get_action(
                    self)
                accel.append(action)

                # lane changing action
                lane_flag = 0
                if type(self.scenario.lanes) is dict:
                    if any(v > 1 for v in self.scenario.lanes.values()):
                        lane_flag = 1
                else:
                    if self.scenario.lanes > 1:
                        lane_flag = 1

                if self.vehicles.get_lane_changing_controller(veh_id) \
                        is not None and lane_flag:
                    new_lane = \
                        self.vehicles.get_lane_changing_controller(
                            veh_id).get_action(self)
                    self.apply_lane_change([veh_id], target_lane=[new_lane])

            self.apply_acceleration(self.controlled_ids, acc=accel)

        # perform (optionally) lane change actions for sumo-controlled
        # human-driven vehicles
        if len(self.sumo_ids) > 0:
            for veh_id in self.sumo_ids:
                # lane changing action
                if self.scenario.lanes > 1:
                    lc_contr = self.vehicles.get_lane_changing_controller(
                        veh_id)
                    if lc_contr is not None:
                        new_lane = lc_contr.get_action(self)
                        self.apply_lane_change([veh_id], target_lane=[new_lane])

        # perform (optionally) routing actions for all vehicle in the network,
        # including rl vehicles
        routing_ids = []
        routing_actions = []
        for veh_id in self.ids:
            if self.vehicles.get_routing_controller(veh_id) is not None:
                routing_ids.append(veh_id)
                route_contr = self.vehicles.get_routing_controller(veh_id)
                routing_actions.append(route_contr.choose_route(self))

        self.choose_routes(veh_ids=routing_ids, route_choices=routing_actions)

        self.apply_rl_actions(rl_actions)

        self.additional_command()

        self.traci_connection.simulationStep()

        # store new observations in the network after traci simulation step
        network_observations = \
            self.traci_connection.vehicle.getSubscriptionResults()
        crash = False
        for veh_id in self.ids:
            prev_pos = self.get_x_by_id(veh_id)
            prev_lane = self.vehicles.get_lane(veh_id)
            try:
                self.vehicles.set_position(
                    veh_id, network_observations[veh_id][tc.VAR_LANEPOSITION])
            except KeyError:
                continue
            self.vehicles.set_edge(
                veh_id, network_observations[veh_id][tc.VAR_ROAD_ID])
            self.vehicles.set_lane(
                veh_id, network_observations[veh_id][tc.VAR_LANE_INDEX])
            if network_observations[veh_id][tc.VAR_LANE_INDEX] != prev_lane \
                    and veh_id in self.rl_ids:
                self.vehicles.set_state(veh_id, "last_lc", self.timer)
            self.vehicles.set_speed(
                veh_id, network_observations[veh_id][tc.VAR_SPEED])

            try:
                change = self.get_x_by_id(veh_id) - prev_pos
                if change < 0:
                    change += self.scenario.length
                new_abs_pos = self.vehicles.get_absolute_position(
                    veh_id) + change
                self.vehicles.set_absolute_position(veh_id, new_abs_pos)
            except ValueError or TypeError:
                self.vehicles.set_absolute_position(veh_id, -1001)

            if self.vehicles.get_position(veh_id) < 0 or \
                            self.vehicles.get_speed(veh_id) < 0:
                crash = True

            # collect headway, leader id, and follower id data
            headway_data = \
                self.get_headway_dict(network_observations=network_observations)

            for veh_id in self.ids:
                try:
                    self.vehicles.set_headway(
                        veh_id, headway_data[veh_id]["headway"])
                    self.vehicles.set_leader(
                        veh_id, headway_data[veh_id]["leader"])
                    self.vehicles.set_follower(
                        veh_id, headway_data[veh_id]["follower"])
                except KeyError:
                    # occurs in the case of crashes, so headway is assumed to
                    # be very small
                    self.vehicles.set_headway(veh_id, 1e-3)
                    self.vehicles.set_leader(veh_id, None)
                    self.vehicles.set_follower(veh_id, None)

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect information of the state of the network based on the
        # environment class used
        if self.vehicles.num_rl_vehicles > 0:
            self.state = self.get_state()
            # rllab requires non-multi agent to have state shape as
            # num-states x num_vehicles
            if not self.multi_agent:
                self.state = self.state.T
        else:
            self.state = []
        # collect observation new state associated with action

        next_observation = list(self.state)

        # crash encodes whether sumo experienced a crash
        crash = \
            crash \
            or self.traci_connection.simulation.getEndingTeleportNumber() != 0 \
            or self.traci_connection.simulation.getStartingTeleportNumber() != 0

        # compute the reward
        if self.vehicles.num_rl_vehicles > 0:
            reward = self.compute_reward(self.state, rl_actions, fail=crash)
        else:
            reward = 0

        # Are we in a multi-agent scenario? If so, the action space is a list.
        if self.multi_agent:
            done_n = self.vehicles.num_rl_vehicles * [0]
            info_n = {'n': []}

            if self.shared_reward:
                info_n['reward_n'] = [reward] * len(self.action_space)
            else:
                info_n['reward_n'] = reward

            if crash:
                done_n = self.vehicles.num_rl_vehicles * [1]
                if self.fail_safe:
                    logging.error("Crash has occurred! Check your failsafes")

            info_n['done_n'] = done_n
            info_n['state'] = self.state
            done = np.all(done_n)
            return self.state, sum(reward), done, info_n

        else:
            if crash:
                logging.info("Crash has occurred! Check failsafes!")
                return next_observation, reward, True, {}
            else:
                return next_observation, reward, False, {}

    # @property
    def _reset(self):
        """
        Resets the state of the environment, and re-initializes the vehicles in
        their starting positions. In "vehicle_arrangement_shuffle" is set to
        True in env_params, the vehicles swap initial positions with one another.
        Also, if a "starting_position_shuffle" is set to True, the initial
        position of vehicles is offset by some value.

        Returns
        -------
        observation: numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # reset the lists of the ids of vehicles in the network
        self.ids = self.vehicles.get_ids()
        self.controlled_ids = self.vehicles.get_controlled_ids()
        self.sumo_ids = self.vehicles.get_sumo_ids()
        self.rl_ids = self.vehicles.get_rl_ids()

        # create the list of colors used to visually distinguish between
        # different types of vehicles
        self.timer = 0
        self.colors = {}
        key_index = 1
        color_choice = np.random.choice(len(COLORS))
        for i in range(self.vehicles.num_types):
            self.colors[self.vehicles.types[i]] = \
                COLORS[(color_choice + key_index) % len(COLORS)]
            key_index += 1

        # perform shuffling (if requested)
        if self.starting_position_shuffle or self.vehicle_arrangement_shuffle:
            if self.starting_position_shuffle:
                x0 = np.random.uniform(0, self.scenario.length)
            else:
                x0 = self.scenario.initial_config.x0

            veh_ids = deepcopy(self.vehicles.get_ids())
            if self.vehicle_arrangement_shuffle:
                random.shuffle(veh_ids)

            initial_positions, initial_lanes = \
                self.scenario.generate_starting_positions(x0=x0)

            initial_state = dict()
            for i, veh_id in enumerate(veh_ids):
                route_id = "route" + initial_positions[i][0]

                # replace initial routes, lanes, and positions to reflect
                # new values
                list_initial_state = list(self.initial_state[veh_id])
                list_initial_state[1] = route_id
                list_initial_state[2] = initial_lanes[i]
                list_initial_state[3] = initial_positions[i][1]
                initial_state[veh_id] = tuple(list_initial_state)

                # replace initial positions in initial observations
                self.initial_observations[veh_id]["edge"] = \
                    initial_positions[i][0]
                self.initial_observations[veh_id]["position"] = \
                    initial_positions[i][1]

            self.initial_state = deepcopy(initial_state)

        # re-initialize the vehicles class with the states of the vehicles at
        # the start of a rollout
        for veh_id in self.ids:
            self.vehicles.set_edge(
                veh_id, self.initial_observations[veh_id]["edge"])
            self.vehicles.set_position(
                veh_id, self.initial_observations[veh_id]["position"])
            self.vehicles.set_lane(
                veh_id, self.initial_observations[veh_id]["lane"])
            self.vehicles.set_speed(
                veh_id, self.initial_observations[veh_id]["speed"])
            self.vehicles.set_route(
                veh_id, self.initial_observations[veh_id]["route"])
            self.vehicles.set_absolute_position(
                veh_id, self.initial_observations[veh_id]["absolute_position"])
            # the time step of the last lane change is always present in the
            # environment, but only used by sub-classes that apply lane changing
            # actions
            self.vehicles.set_state(veh_id, "last_lc",
                                    -1 * self.lane_change_duration)

        # re-initialize memory on last lc
        self.prev_last_lc = dict()
        for veh_id in self.ids:
            self.prev_last_lc[veh_id] = self.vehicles.get_state(veh_id,
                                                                "last_lc")

        # reset the list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        for veh_id in self.ids:
            type_id, route_id, lane_index, lane_pos, speed, pos = \
                self.initial_state[veh_id]

            # clear controller acceleration queue of traci-controlled vehicles
            if veh_id in self.controlled_ids:
                self.vehicles.get_acc_controller(veh_id).reset_delay(self)

            # clear vehicles from traci connection and re-introduce vehicles
            # with pre-defined initial position
            try:
                self.traci_connection.vehicle.remove(veh_id)
                self.traci_connection.vehicle.addFull(
                    veh_id, route_id, typeID=str(type_id),
                    departLane=str(lane_index),
                    departPos=str(lane_pos), departSpeed=str(speed))
            except:
                # in case a vehicle is removed via a crash
                self.traci_connection.vehicle.addFull(
                    veh_id, route_id, typeID=str(type_id),
                    departLane=str(lane_index),
                    departPos=str(lane_pos), departSpeed=str(speed))

            self.traci_connection.vehicle.setColor(
                veh_id, self.colors[self.vehicles.get_state(veh_id, 'type')])

            # set top speed
            self.traci_connection.vehicle.setMaxSpeed(veh_id, self.max_speed)

            # reset speed mode
            self.set_speed_mode(veh_id)

            # reset lane change mode
            self.set_lane_change_mode(veh_id)

        self.traci_connection.simulationStep()

        for veh_id in self.ids:
            # collect headway, leader id, and follower id data
            headway = self.traci_connection.vehicle.getLeader(veh_id, 200)
            if headway is None:
                self.vehicles.set_leader(veh_id, None)
                self.vehicles.set_headway(veh_id, 1e-3)
            else:
                self.vehicles.set_leader(veh_id, headway[0])
                self.vehicles.set_headway(veh_id, headway[1])
                self.vehicles.set_follower(headway[0], veh_id)

        if self.multi_agent:
            self.state = self.get_state()
        else:
            self.state = self.get_state().T

        observation = list(self.state)
        return observation

    def additional_command(self):
        """
        Additional commands that may be performed before a simulation step.
        """
        pass

    def apply_rl_actions(self, rl_actions):
        """
        Specifies the actions to be performed by rl_vehicles

        Parameters
        ----------
        rl_actions: numpy ndarray
            list of actions provided by the RL algorithm
        """
        pass

    def apply_acceleration(self, veh_ids, acc):
        """
        Applies the acceleration requested by a vehicle in sumo. Note that, if
        the sumo-specified speed mode of the vehicle is not "aggressive", the
        acceleration may be clipped by some saftey velocity or maximum possible
        acceleration.

        Parameters
        ----------
        veh_ids: list of strings
            vehicles IDs associated with the requested accelerations
        acc: numpy array or list of float
            requested accelerations from the vehicles
        """
        for i, veh_id in enumerate(veh_ids):
            acc_contr = self.vehicles.get_acc_controller(veh_id)
            if self.fail_safe == 'instantaneous':
                safe_acc = acc_contr.get_safe_action_instantaneous(self, acc[i])
            elif self.fail_safe == 'safe_velocity':
                safe_acc = acc_contr.get_safe_action(self, acc[i])
            else:
                # Test for multi-agent
                if self.multi_agent and (veh_id in self.rl_ids):
                    safe_acc = acc[i][0]
                else:
                    safe_acc = acc[i]

            if self.multi_agent and (veh_id in self.rl_ids):
                acc[i][0] = safe_acc
            else:
                acc[i] = safe_acc

        # issue traci command for requested acceleration
        this_vel = np.array(self.vehicles.get_speed(veh_ids))
        if self.multi_agent and (veh_id in self.rl_ids):
            acc_arr = np.asarray([element2 for elem in acc for element in elem
                                  for element2 in element])
        else:
            acc_arr = np.array(acc)

        requested_next_vel = this_vel + acc_arr * self.time_step
        actual_next_vel = requested_next_vel.clip(min=0)

        for i, vid in enumerate(veh_ids):
            self.traci_connection.vehicle.slowDown(vid, actual_next_vel[i], 1)

    def apply_lane_change(self, veh_ids, direction=None, target_lane=None):
        """
        Applies an instantaneous lane-change to a set of vehicles, while
        preventing vehicles from moving to lanes that do not exist.

        Parameters
        ----------
        veh_ids: list of strings
            vehicles IDs associated with the requested accelerations
        direction: list of int (-1, 0, or 1), optional
            -1: lane change to the right
             0: no lange change
             1: lane change to the left
        target_lane: list of int, optional
            lane indices the vehicles should lane-change to in the next step

        Raises
        ------
        ValueError
            If either both or none of "direction" and "target_lane" are provided
            as inputs. Only one should be provided at a time.
        ValueError
            If any of the direction values are not -1, 0, or 1.
        """
        if direction is not None and target_lane is not None:
            raise ValueError("Cannot provide both a direction and target_lane.")
        elif direction is None and target_lane is None:
            raise ValueError("A direction or target_lane must be specified.")

        for veh_id in veh_ids:
            self.prev_last_lc[veh_id] = self.vehicles.get_state(veh_id,
                                                                "last_lc")

        if self.scenario.lanes == 1:
            logging.error("Uh oh, single lane track.")
            return -1

        current_lane = np.array(self.vehicles.get_lane(veh_ids))

        if target_lane is None:
            # if any of the directions are not -1, 0, or 1, raise a ValueError
            if np.any(np.sign(direction) != np.array(direction)):
                raise ValueError("Direction values for lane changes may only "
                                 "be: -1, 0, or 1.")

            target_lane = current_lane + np.array(direction)

        target_lane = np.clip(target_lane, 0, self.scenario.lanes - 1)

        for i, vid in enumerate(veh_ids):
            if vid in self.rl_ids:
                if target_lane[i] != current_lane[i]:
                    self.traci_connection.vehicle.changeLane(vid, int(
                        target_lane[i]), 100000)
            else:
                self.traci_connection.vehicle.changeLane(vid,
                                                         int(target_lane[i]),
                                                         100000)

    def choose_routes(self, veh_ids, route_choices):
        """
        Updates the route choice of vehicles in the network.

        Parameters
        ----------
        veh_ids: list
            list of vehicle identifiers
        route_choices: numpy array or list of floats
            list of edges the vehicle wishes to traverse, starting with the edge
            the vehicle is currently on. If a value of None is provided, the
            vehicle does not update its route
        """
        for i, veh_id in enumerate(veh_ids):
            if route_choices[i] is not None:
                self.traci_connection.vehicle.setRoute(
                    vehID=veh_id, edgeList=route_choices[i])
                self.vehicles.set_route(veh_id, route_choices[i])

    def set_speed_mode(self, veh_id):
        """
        Sets the SUMO-defined speed mode with Traci. This is used to constrain
        acceleration actions.

        The available speed modes are as follows:
         - "no_collide" (default): Human and RL cars are preventing from
           reaching speeds that may cause crashes (also serves as a failsafe).
         - "aggressive": Human and RL cars are not limited by sumo with regard
           to their accelerations, and can crash longitudinally

        Parameters
        ----------
        veh_id: string
            vehicle identifier
        """
        speed_mode_id = 1
        if veh_id in self.rl_ids:
            speed_mode = self.sumo_params.rl_speed_mode
        else:
            speed_mode = self.sumo_params.human_speed_mode

        if speed_mode == "aggressive":
            speed_mode_id = 0
        elif speed_mode == "no_collide":
            speed_mode_id = 1
        else:
            logging.error("Invalid Speed Mode!! Defaulting to no collision.")

        self.traci_connection.vehicle.setSpeedMode(veh_id, speed_mode_id)

    def set_lane_change_mode(self, veh_id):
        """
        Sets the SUMO-defined lane-change mode with Traci. This is used to
        constrain lane-changing actions.

        The available lane-changing modes are as follows:
         - default: Human and RL cars can only safely change into lanes
         - "strategic": Human cars make lane changes in accordance with SUMO to
           provide speed boosts
         - "no_lat_collide": RL cars can lane change into any space, no matter
           how likely it is to crash
         - "aggressive": RL cars are not limited by sumo with regard to their
           lane-change actions, and can crash longitudinally

        Parameters
        ----------
        veh_id: string
            vehicle identifier
        """
        lc_mode_id = 256

        if veh_id in self.rl_ids:
            lc_mode = self.sumo_params.rl_lane_change_mode
        else:
            lc_mode = self.sumo_params.human_lane_change_mode

        if lc_mode == "aggressive":
            lc_mode_id = 0
        elif lc_mode == "no_lat_collide":
            lc_mode_id = 256
        elif lc_mode == "strategic":
            lc_mode_id = 853
        else:
            logging.error("Invalid Speed Mode!!")

        self.traci_connection.vehicle.setLaneChangeMode(veh_id, lc_mode_id)

    def get_x_by_id(self, veh_id):
        """
        Provides a 1-dimensional representation of the position of a vehicle
        in the network.

        Parameters
        ----------
        veh_id: string
            vehicle identifier

        Yields
        ------
        float
            position of a vehicle relative to a certain reference.
        """
        if self.vehicles.get_edge(veh_id) == '':
            # occurs when a vehicle crashes is teleported for some other reason
            return 0.
        return self.scenario.get_x(self.vehicles.get_edge(veh_id),
                                   self.vehicles.get_position(veh_id))

    def sort_by_position(self):
        """
        Sorts the vehicle ids of vehicles in the network by position.
        The base environment does this by sorting vehicles by their absolute
        position, as specified by the "get_x_by_id" function.

        Returns
        -------
        sorted_ids: list
            a list of all vehicle IDs sorted by position
        sorted_extra_data: list or tuple
            an extra component (list, tuple, etc...) containing extra sorted
            data, such as positions. If no extra component is needed, a value
            of None should be returned
        """
        sorted_indx = np.argsort(self.vehicles.get_absolute_position(self.ids))
        sorted_ids = np.array(self.ids)[sorted_indx]
        return sorted_ids, None

    def get_headway_dict(self, network_observations):
        """
        Collects the headways, leaders, and followers of all vehicles at once.
        The base environment does by using traci calls.

        Parameters
        ----------
        network_observations: dictionary
            key = vehicle IDs
            elements = variable state properties of the vehicle (including
            headway)

        Yields
        ------
        dictionary
            key = vehicle IDs
            elements = headway, leader id, and follower id for the vehicle
        """
        vehicles = dict.fromkeys(self.ids)

        for veh_id in self.ids:
            vehicles[veh_id] = dict()

        for veh_id in self.ids:
            try:
                headway = network_observations[veh_id][tc.VAR_LEADER]
                if headway is None:
                    vehicles[veh_id]["leader"] = None
                    vehicles[veh_id]["follower"] = None
                    vehicles[veh_id]["headway"] = 1e-3
                else:
                    vehicles[veh_id]["headway"] = headway[1]
                    vehicles[veh_id]["leader"] = headway[0]
                    vehicles[headway[0]]["follower"] = veh_id
            except KeyError:
                # this is used to deal with the absence of network observations
                # upon reset. It only applies for the very first time step
                vehicles[veh_id]["leader"] = None
                vehicles[veh_id]["follower"] = None
                vehicles[veh_id]["headway"] = 1e-3

        return vehicles

    def get_state(self):
        """
        Returns the state of the simulation as perceived by the learning agent.
        MUST BE implemented in new environments.

        Returns
        -------
        state: numpy ndarray
            information on the state of the vehicles, which is provided to the
            agent
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Identifies the dimensions and bounds of the action space (needed for
        rllab environments).
        MUST BE implemented in new environments.

        Yields
        -------
        rllab Box or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Identifies the dimensions and bounds of the observation space (needed
        for rllab environments).
        MUST BE implemented in new environments.

        Yields
        -------
        rllab Box or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        Reward function for RL.
        MUST BE implemented in new environments.

        Parameters
        ----------
        state: numpy ndarray
            state of all the vehicles in the simulation
        rl_actions: numpy ndarray
            actions performed by rl vehicles
        kwargs: dictionary
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward: float
        """
        raise NotImplementedError

    def terminate(self):
        """
        Closes the TraCI I/O connection. Should be done at end of every
        experiment. Must be in Environment because the environment opens the
        TraCI connection.
        """
        self._close()

    def _close(self):
        self.traci_connection.close()

    def _seed(self, seed=None):
        return []
