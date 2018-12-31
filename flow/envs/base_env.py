"""Base environment class. This is the parent of all other environments."""

from copy import deepcopy
import gym
from gym.spaces import Box
import logging
import os
import sys
import time
import traceback
import numpy as np
import random
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer

from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException

import sumolib

try:
    # Import serializable if rllab is installed
    from rllab.core.serializable import Serializable
    serializable_flag = True
except ImportError:
    serializable_flag = False

from flow.core.util import ensure_dir
from flow.core.kernel import Kernel

# Number of retries on restarting SUMO before giving up
RETRIES_ON_ERROR = 10

# pick out the correct class definition
if serializable_flag:
    classdef = (gym.Env, Serializable)
else:
    classdef = (gym.Env,)

# colors for vehicles
WHITE = (255, 255, 255, 255)
CYAN = (0, 255, 255, 255)
RED = (255, 0, 0, 255)


class Env(*classdef):
    """Base environment class.

    Provides the interface for controlling a SUMO simulation. Using this
    class, you can start sumo, provide a scenario to specify a
    configuration and controllers, perform simulation steps, and reset the
    simulation to an initial configuration.

    Env is Serializable to allow for pickling and replaying of the policy.

    This class cannot be used as is: you must extend it to implement an
    action applicator method, and properties to define the MDP if you
    choose to use it with an rl library (e.g. RLlib). This can be done by
    overloading the following functions in a child class:
     - action_space
     - observation_space
     - apply_rl_action
     - get_state
     - compute_reward

    Attributes
    ----------
    env_params : flow.core.params.EnvParams
       see flow/core/params.py
    sim_params: flow.core.params.SimParams
       see flow/core/params.py
    scenario: Scenario type
        see flow/scenarios/base_scenario.py
    """

    def __init__(self, env_params, sim_params, scenario):
        # Invoke serializable if using rllab

        if serializable_flag:
            Serializable.quick_init(self, locals())

        self.env_params = env_params
        self.scenario = scenario
        self.sim_params = sim_params
        time_stamp = ''.join(str(time.time()).split('.'))
        if os.environ.get("TEST_FLAG", 0):
            # 1.0 works with stress_test_start 10k times
            time.sleep(1.0 * int(time_stamp[-6:]) / 1e6)
        # FIXME: this is sumo-specific
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()
        self.vehicles = scenario.vehicles
        self.traffic_lights = scenario.traffic_lights
        # time_counter: number of steps taken since the start of a rollout
        self.time_counter = 0
        # step_counter: number of total steps taken
        self.step_counter = 0
        # initial_state:
        #   Key = Vehicle ID,
        #   Entry = (type_id, route_id, lane_index, lane_pos, speed, pos)
        self.initial_state = {}
        self.state = None
        self.obs_var_labels = []

        # simulation step size
        self.sim_step = sim_params.sim_step

        # the available_routes variable contains a dictionary of routes
        # vehicles can traverse; to be used when routes need to be chosen
        # dynamically
        self.available_routes = self.scenario.rts

        # TraCI connection used to communicate with sumo
        self.traci_connection = None

        # dictionary of initial observations used while resetting vehicles
        # after each rollout
        self.initial_observations = dict.fromkeys(self.vehicles.get_ids())

        # store the initial vehicle ids
        self.initial_ids = deepcopy(self.vehicles.get_ids())

        # store the initial state of the vehicles class (for restarting sumo)
        self.initial_vehicles = deepcopy(self.vehicles)

        # colors used to distinguish between types of vehicles in the network
        self.colors = {}

        # create the Flow kernel
        self.k = Kernel(simulator="traci", sim_params=sim_params)

        # initialize the simulation using the simulation kernel. This will use
        # the scenario kernel as an input in order to determine what network
        # needs to be simulated.
        self.traci_connection = self.k.simulation.start_simulation(
            scenario=self.scenario, sim_params=sim_params)

        # pass the kernel api to the kernel and it's subclasses
        self.k.pass_api(self.traci_connection)

        self.setup_initial_state()

        # use pyglet to render the simulation
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sim_params.save_render
            sight_radius = self.sim_params.sight_radius
            pxpm = self.sim_params.pxpm
            show_radius = self.sim_params.show_radius

            # get network polygons
            network = []
            for lane_id in self.traci_connection.lane.getIDList():
                _lane_poly = self.traci_connection.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)

            # instantiate a pyglet renderer
            self.renderer = Renderer(
                network,
                self.sim_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)

            # render a frame
            self.render(reset=True)
        elif self.sim_params.render in [True, False]:
            pass  # default to sumo-gui (if True) or sumo (if False)
        else:
            raise ValueError("Mode %s is not supported!" %
                             self.sim_params.render)

    def restart_simulation(self, sim_params, render=None):
        """Restart an already initialized simulation instance.

        This is used when visualizing a rollout, in order to update the
        rendering with potentially a gui and export emission data from sumo.

        This is also used to handle cases when the runtime of an experiment is
        too long, causing the sumo instance

        Parameters
        ----------
        sim_params : flow.core.params.SimParams
            sumo-specific parameters
        render: bool, optional
            specifies whether to use the gui
        """
        self.traci_connection.close(False)
        self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sim_params.render = render

        if sim_params.emission_path is not None:
            ensure_dir(sim_params.emission_path)
            self.sim_params.emission_path = sim_params.emission_path

        self.traci_connection = self.k.simulation.start_simulation(
            scenario=self.scenario, sim_params=self.sim_params)
        self.k.pass_api(self.traci_connection)
        self.setup_initial_state()

    def setup_initial_state(self):
        """Store information on the initial state of vehicles in the network.

        This information is to be used upon reset. This method also adds this
        information to the self.vehicles class and starts a subscription with
        sumo to collect state information each step.
        """
        # determine whether to shuffle the vehicles
        if self.scenario.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        # generate starting position for vehicles in the network
        start_pos, start_lanes = self.scenario.generate_starting_positions(
            num_vehicles=len(self.initial_ids))

        # save the initial state. This is used in the _reset function
        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.scenario.vehicles.get_type(veh_id)
            pos = start_pos[i][1]
            lane = start_lanes[i]
            speed = self.scenario.vehicles.get_initial_speed(veh_id)
            route_id = "route" + start_pos[i][0]

            self.initial_state[veh_id] = (type_id, route_id, lane, pos, speed)

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

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
        done: bool
            indicates whether the episode has ended
        info: dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.vehicles.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.vehicles.get_controlled_ids():
                    accel_contr = self.vehicles.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.apply_acceleration(self.vehicles.get_controlled_ids(),
                                        accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.vehicles.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.vehicles.get_controlled_lc_ids():
                    lc_contr = self.vehicles.get_lane_changing_controller(
                        veh_id)
                    target_lane = lc_contr.get_action(self)
                    direction.append(target_lane)
                self.apply_lane_change(
                    self.vehicles.get_controlled_lc_ids(), direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.vehicles.get_ids():
                if self.vehicles.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.vehicles.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # collect subscription information from sumo
            vehicle_obs = \
                self.traci_connection.vehicle.getSubscriptionResults()
            id_lists = \
                self.traci_connection.simulation.getSubscriptionResults()

            # store new observations in the vehicles and traffic lights class
            self.vehicles.update(vehicle_obs, id_lists, self)

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            self.update_vehicle_colors()

            # collect list of sorted vehicle ids
            self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()
        if isinstance(states, dict):
            self.state = {}
            next_observation = {}
            done = {}
            infos = {}
            temp_state = states
            for key, state in temp_state.items():
                # collect information of the state of the network based on the
                # environment class used
                self.state[key] = np.asarray(state).T

                # collect observation new state associated with action
                next_observation[key] = np.copy(self.state[key])

                # test if a crash has occurred
                done[key] = crash
                # test if the agent has exited the system, if so
                # its agent should be done
                # FIXME(ev) this assumes that agents are single vehicles
                if key in self.vehicles.get_arrived_ids():
                    done[key] = True
                # check if an agent is done
                if crash:
                    done['__all__'] = True
                else:
                    done['__all__'] = False
                infos[key] = {}
        else:
            # collect information of the state of the network based on the
            # environment class used
            self.state = np.asarray(states).T

            # collect observation new state associated with action
            next_observation = np.copy(states)

            # test if the agent should terminate due to a crash
            done = crash

            # compute the info for each agent
            infos = {}

        # compute the reward
        rl_clipped = self.clip_actions(rl_actions)
        reward = self.compute_reward(rl_clipped, fail=crash)

        return next_observation, reward, done, infos

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment, and re-initializes the vehicles in their starting
        positions.

        If "shuffle" is set to True in InitialConfig, the initial positions of
        vehicles is recalculated and the vehicles are shuffled.

        Returns
        -------
        observation: numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # reset the time counter
        self.time_counter = 0

        # warn about not using restart_instance when using inflows
        if len(self.scenario.net_params.inflows.get()) > 0 and \
                not self.sim_params.restart_instance:
            print(
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "WARNING: Inflows will cause computational performance to\n"
                "significantly decrease after large number of rollouts. In \n"
                "order to avoid this, set SumoParams(restart_instance=True).\n"
                "**********************************************************\n"
                "**********************************************************\n"
                "**********************************************************"
            )

        if self.sim_params.restart_instance or self.step_counter > 2e6:
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sim_params.seed = random.randint(0, 1e5)
            # modify the vehicles class to match initial data
            self.vehicles = deepcopy(self.initial_vehicles)
            # restart the simulation instance
            self.restart_simulation(self.sim_params)

        elif self.scenario.initial_config.shuffle:
            # perform shuffling (if requested)
            self.setup_initial_state()

        # clear all vehicles from the network and the vehicles class
        for veh_id in self.traci_connection.vehicle.getIDList():
            try:
                self.traci_connection.vehicle.remove(veh_id)
                self.traci_connection.vehicle.unsubscribe(veh_id)
                self.vehicles.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.vehicles.get_ids()):
            self.vehicles.remove(veh_id)
            # do not try to remove the vehicles from the network in the first
            # step after initializing the network, as there will be no vehicles
            if self.step_counter == 0:
                continue
            try:
                self.traci_connection.vehicle.remove(veh_id)
                self.traci_connection.vehicle.unsubscribe(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # reintroduce the initial vehicles to the network
        for veh_id in self.initial_ids:
            type_id, route_id, lane_index, pos, speed = \
                self.initial_state[veh_id]

            try:
                self.traci_connection.vehicle.addFull(
                    veh_id,
                    route_id,
                    typeID=str(type_id),
                    departLane=str(lane_index),
                    departPos=str(pos),
                    departSpeed=str(speed))
            except (FatalTraCIError, TraCIException):
                # if a vehicle was not removed in the first attempt, remove it
                # now and then reintroduce it
                self.traci_connection.vehicle.remove(veh_id)
                self.traci_connection.vehicle.addFull(
                    veh_id,
                    route_id,
                    typeID=str(type_id),
                    departLane=str(lane_index),
                    departPos=str(pos),
                    departSpeed=str(speed))

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # collect subscription information from sumo
        vehicle_obs = self.traci_connection.vehicle.getSubscriptionResults()
        id_lists = self.traci_connection.simulation.getSubscriptionResults()

        # store new observations in the vehicles and traffic lights class
        self.vehicles.update(vehicle_obs, id_lists, self)

        # store new observations in the vehicles and traffic lights class
        self.k.update(reset=True)

        # update the colors of vehicles
        self.update_vehicle_colors()

        # check to make sure all vehicles have been spawned
        if len(self.initial_ids) > self.vehicles.num_vehicles:
            missing_vehicles = list(
                set(self.initial_ids) - set(self.vehicles.get_ids()))
            logging.error('Not enough vehicles have spawned! Bad start?')
            logging.error('Missing vehicles / initial state:')
            for veh_id in missing_vehicles:
                logging.error('- {}: {}'.format(veh_id,
                                                self.initial_state[veh_id]))
            sys.exit()

        self.prev_last_lc = dict()
        for veh_id in self.vehicles.get_ids():
            # re-initialize the vehicles class with the states of the vehicles
            # at the start of a rollout
            self.vehicles.set_absolute_position(veh_id,
                                                self.get_x_by_id(veh_id))

            # re-initialize memory on last lc
            self.prev_last_lc[veh_id] = -float("inf")

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        states = self.get_state()
        if isinstance(states, dict):
            self.state = {}
            observation = {}
            for key, state in states.items():
                # collect information of the state of the network based on the
                # environment class used
                self.state[key] = np.asarray(state).T

                # collect observation new state associated with action
                observation[key] = np.copy(self.state[key]).tolist()

        else:
            # collect information of the state of the network based on the
            # environment class used
            self.state = np.asarray(states).T

            # observation associated with the reset (no warm-up steps)
            observation = np.copy(states)

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)

        # render a frame
        self.render(reset=True)

        return observation

    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        pass

    def clip_actions(self, rl_actions=None):
        """Clip the actions passed from the RL agent.

        Parameters
        ----------
        rl_actions : list or numpy ndarray
            list of actions provided by the RL algorithm

        Returns
        -------
        numpy ndarray (float)
            The rl_actions clipped according to the box
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return None

        # clip according to the action space requirements
        if isinstance(self.action_space, Box):
            rl_actions = np.clip(
                rl_actions,
                a_min=self.action_space.low,
                a_max=self.action_space.high)
        return rl_actions

    def apply_rl_actions(self, rl_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by SUMO.

        Parameters
        ----------
        rl_actions : list or numpy ndarray
            list of actions provided by the RL algorithm
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        rl_clipped = self.clip_actions(rl_actions)
        self._apply_rl_actions(rl_clipped)

    def _apply_rl_actions(self, rl_actions):
        raise NotImplementedError

    def apply_acceleration(self, veh_ids, acc):
        """Apply the acceleration requested by a vehicle in SUMO.

        Note that, if the SUMO-specified speed mode of the vehicle is not
        "aggressive", the acceleration may be clipped by some safety velocity
        or maximum possible acceleration.

        Parameters
        ----------
        veh_ids: list of str
            vehicles IDs associated with the requested accelerations
        acc: numpy ndarray or list of float
            requested accelerations from the vehicles
        """
        for i, vid in enumerate(veh_ids):
            if acc[i] is not None:
                this_vel = self.vehicles.get_speed(vid)
                next_vel = max([this_vel + acc[i] * self.sim_step, 0])
                self.traci_connection.vehicle.slowDown(vid, next_vel, 1)

    def apply_lane_change(self, veh_ids, direction):
        """Apply an instantaneous lane-change to a set of vehicles.

        This method also prevents vehicles from moving to lanes that do not
        exist, and set the "last_lc" variable for RL vehicles that lane changed
        to match the current time step, in order to assist in maintaining a
        lane change duration for these vehicles.

        Parameters
        ----------
        veh_ids: list of str
            vehicles IDs associated with the requested accelerations
        direction: list of {-1, 0, 1}
            -1: lane change to the right
             0: no lane change
             1: lane change to the left

        Raises
        ------
        ValueError
            If any of the direction values are not -1, 0, or 1.
        """
        # if any of the directions are not -1, 0, or 1, raise a ValueError
        if any(d not in [-1, 0, 1] for d in direction):
            raise ValueError(
                "Direction values for lane changes may only be: -1, 0, or 1.")

        for i, veh_id in enumerate(veh_ids):
            # check for no lane change
            if direction[i] == 0:
                continue

            # compute the target lane, and clip it so vehicle don't try to lane
            # change out of range
            this_lane = self.vehicles.get_lane(veh_id)
            this_edge = self.vehicles.get_edge(veh_id)
            target_lane = min(
                max(this_lane + direction[i], 0),
                self.scenario.num_lanes(this_edge) - 1)

            # perform the requested lane action action in TraCI
            if target_lane != this_lane:
                self.traci_connection.vehicle.changeLane(
                    veh_id, int(target_lane), 100000)

                if veh_id in self.vehicles.get_rl_ids():
                    self.prev_last_lc[veh_id] = \
                        self.vehicles.get_state(veh_id, "last_lc")

    def choose_routes(self, veh_ids, route_choices):
        """Update the route choice of vehicles in the network.

        Parameters
        ----------
        veh_ids: list
            list of vehicle identifiers
        route_choices: numpy array or list of floats
            list of edges the vehicle wishes to traverse, starting with the
            edge the vehicle is currently on. If a value of None is provided,
            the vehicle does not update its route
        """
        for i, veh_id in enumerate(veh_ids):
            if route_choices[i] is not None:
                self.traci_connection.vehicle.setRoute(
                    vehID=veh_id, edgeList=route_choices[i])

    def get_x_by_id(self, veh_id):
        """Provide a 1-D representation of the position of a vehicle.

        Note: These values are only meaningful if the specify_edge_starts
        method in the scenario is set appropriately; otherwise, a value of 0 is
        returned for all vehicles.

        Parameters
        ----------
        veh_id: str
            vehicle identifier

        Returns
        -------
        float
            position of a vehicle relative to a certain reference.
        """
        if self.vehicles.get_edge(veh_id) == '':
            # occurs when a vehicle crashes is teleported for some other reason
            return 0.
        return self.scenario.get_x(
            self.vehicles.get_edge(veh_id), self.vehicles.get_position(veh_id))

    def sort_by_position(self):
        """Sort the vehicle ids of vehicles in the network by position.

        The base environment does this by sorting vehicles by their absolute
        position.

        Returns
        -------
        sorted_ids: list <str>
            a list of all vehicle IDs sorted by position
        sorted_extra_data: list or tuple
            an extra component (list, tuple, etc...) containing extra sorted
            data, such as positions. If no extra component is needed, a value
            of None should be returned
        """
        if self.env_params.sort_vehicles:
            sorted_ids = sorted(
                self.vehicles.get_ids(),
                key=self.vehicles.get_absolute_position)
            return sorted_ids, None
        else:
            return self.vehicles.get_ids(), None

    def update_vehicle_colors(self):
        """Modify the color of vehicles if rendering is active.

        The colors of all vehicles are updated as follows:
        - red: autonomous (rl) vehicles
        - white: unobserved human-driven vehicles
        - cyan: observed human-driven vehicles
        """
        # do not change the colors of vehicles if the sumo-gui is not active
        # (in order to avoid slow downs)
        if self.sim_params.render is not True:
            return

        for veh_id in self.vehicles.get_rl_ids():
            try:
                # color rl vehicles red
                self.traci_connection.vehicle.setColor(
                    vehID=veh_id, color=RED)
            except (FatalTraCIError, TraCIException):
                pass

        for veh_id in self.vehicles.get_human_ids():
            try:
                if veh_id in self.vehicles.get_observed_ids():
                    # color observed human-driven vehicles cyan
                    color = CYAN
                else:
                    # color unobserved human-driven vehicles white
                    color = WHITE
                self.traci_connection.vehicle.setColor(
                    vehID=veh_id, color=color)
            except (FatalTraCIError, TraCIException):
                pass

        # clear the list of observed vehicles
        for veh_id in self.vehicles.get_observed_ids():
            self.vehicles.remove_observed(veh_id)

    def get_state(self):
        """Return the state of the simulation as perceived by the RL agent.

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
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def compute_reward(self, rl_actions, **kwargs):
        """Reward function for the RL agent(s).

        MUST BE implemented in new environments.
        Defaults to 0 for non-implemented environments.

        Parameters
        ----------
        rl_actions: numpy ndarray
            actions performed by rl vehicles
        kwargs: dict
            other parameters of interest. Contains a "fail" element, which
            is True if a vehicle crashed, and False otherwise

        Returns
        -------
        reward: float or list <float>
        """
        return 0

    def terminate(self):
        """Close the TraCI I/O connection.

        Should be done at end of every experiment. Must be in Env because the
        environment opens the TraCI connection.
        """
        print(
            "Closing connection to TraCI and stopping simulation.\n"
            "Note, this may print an error message when it closes."
        )
        self.traci_connection.close()
        self.scenario.close()

        # close pyglet renderer
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            self.renderer.close()

    def render(self, reset=False, buffer_length=5):
        """Render a frame.

        Parameters
        ----------
        reset: bool
            set to True to reset the buffer
        buffer_length: int
            length of the buffer
        """
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    self.frame_buffer.append(self.frame.copy())
                    self.sights_buffer.append(self.sights.copy())
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)

    def pyglet_render(self):
        """Render a frame using pyglet."""

        # get human and RL simulation status
        human_idlist = self.vehicles.get_human_ids()
        machine_idlist = self.vehicles.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.scenario.max_speed
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.vehicles.get_timestep(id),
                     self.vehicles.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.vehicles.get_orientation(id))
                machine_dynamics.append(
                    self.vehicles.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.vehicles.get_timestep(id),
                     self.vehicles.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.vehicles.get_orientation(id))
                human_dynamics.append(
                    self.vehicles.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.vehicles.get_timestep(id),
                 self.vehicles.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.vehicles.get_orientation(id))
            machine_dynamics.append(
                self.vehicles.get_speed(id)/max_speed)

        # step the renderer
        self.frame = self.renderer.render(human_orientations,
                                          machine_orientations,
                                          human_dynamics,
                                          machine_dynamics,
                                          human_logs,
                                          machine_logs)

        # get local observation of RL vehicles
        self.sights = []
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if "track" in id:
                orientation = self.vehicles.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.vehicles.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)
