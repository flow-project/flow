"""Base environment class. This is the parent of all other environments."""

import logging
import os
import sys
from copy import deepcopy
import time
import traceback
import numpy as np
import random
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer

import gym
from gym.spaces import Box
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

# pick out the correct class definition
if serializable_flag:
    classdef = (gym.Env, Serializable)
else:
    classdef = (gym.Env,)


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
    env_params: EnvParams type:
       see flow/core/params.py
    sumo_params: SumoParams type
       see flow/core/params.py
    scenario: Scenario type
        see flow/scenarios/base_scenario.py
    """

    def __init__(self, env_params, sumo_params, scenario):
        # Invoke serializable if using rllab
        if serializable_flag:
            Serializable.quick_init(self, locals())

        self.env_params = env_params
        self.scenario = scenario
        self.net_params = scenario.net_params
        self.initial_config = scenario.initial_config
        self.sumo_params = sumo_params
        time_stamp = ''.join(str(time.time()).split('.'))
        if os.environ.get("TEST_FLAG", 0):
            # 1.0 works with stress_test_start 10k times
            time.sleep(1.0 * int(time_stamp[-6:]) / 1e6)
        self.sumo_params.port = sumolib.miscutils.getFreeSocketPort()
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
        self.sim_step = sumo_params.sim_step

        self.vehicle_arrangement_shuffle = \
            env_params.vehicle_arrangement_shuffle
        self.starting_position_shuffle = env_params.starting_position_shuffle

        # the simulator used by this environment
        self.simulator = 'traci'

        # create the Flow kernel
        self.k = Kernel(simulator=self.simulator,
                        sim_params=self.sumo_params)

        # use the scenario class's network parameters to generate the necessary
        # scenario components within the scenario kernel
        self.k.scenario.generate_network(scenario)

        # initial the vehicles kernel using the VehicleParams object
        self.k.vehicle.initialize(deepcopy(scenario.vehicles))

        # initialize the simulation using the simulation kernel. This will use
        # the scenario kernel as an input in order to determine what network
        # needs to be simulated.
        self.traci_connection = self.k.simulation.start_simulation(
            scenario=self.k.scenario, sim_params=sumo_params)

        # pass the kernel api to the kernel and it's subclasses
        self.k.pass_api(self.traci_connection)

        # the available_routes variable contains a dictionary of routes
        # vehicles can traverse; to be used when routes need to be chosen
        # dynamically
        self.available_routes = self.k.scenario.rts

        # store the initial vehicle ids
        self.initial_ids = deepcopy(scenario.vehicles.ids)

        # store the initial state of the vehicles class (for restarting sumo)
        self.k.vehicle.kernel_api = None
        self.k.vehicle.master_kernel = None
        self.initial_vehicles = deepcopy(self.k.vehicle)
        self.k.vehicle.kernel_api = self.traci_connection
        self.k.vehicle.master_kernel = self.k

        self.setup_initial_state()

        # use pyglet to render the simulation
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sumo_params.save_render
            sight_radius = self.sumo_params.sight_radius
            pxpm = self.sumo_params.pxpm
            show_radius = self.sumo_params.show_radius

            # get network polygons
            network = []
            for lane_id in self.traci_connection.lane.getIDList():
                _lane_poly = self.traci_connection.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)

            # instantiate a pyglet renderer
            self.renderer = Renderer(
                network,
                self.sumo_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)

            # render a frame
            self.render(reset=True)
        elif self.sumo_params.render in [True, False]:
            pass  # default to sumo-gui (if True) or sumo (if False)
        else:
            raise ValueError('Mode %s is not supported!' %
                             self.sumo_params.render)

    # TODO(ak): Rename to restart_simulation
    def restart_sumo(self, sumo_params, render=None):
        """Restart an already initialized sumo instance.

        This is used when visualizing a rollout, in order to update the
        rendering with potentially a gui and export emission data from sumo.

        This is also used to handle cases when the runtime of an experiment is
        too long, causing the sumo instance

        Parameters
        ----------
        sumo_params: SumoParams type  # TODO: make ambiguous
            sumo-specific parameters
        render: bool, optional
            specifies whether to use sumo's gui
        """
        self.k.close()

        # killed the sumo process if using sumo/TraCI
        if self.simulator == 'traci':
            self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sumo_params.render = render

        if sumo_params.emission_path is not None:
            ensure_dir(sumo_params.emission_path)
            self.sumo_params.emission_path = sumo_params.emission_path

        self.k.scenario.generate_network(self.scenario)
        self.k.vehicle.initialize(deepcopy(self.scenario.vehicles))
        self.traci_connection = self.k.simulation.start_simulation(
            scenario=self.k.scenario, sim_params=self.sumo_params)
        self.k.pass_api(self.traci_connection)

        self.setup_initial_state()

    def setup_initial_state(self):
        """Return information on the initial state of vehicles in the network.

        This information is to be used upon reset.

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
        # store new observations in the vehicles and traffic lights class
        self.k.update(reset=True)

        # check to make sure all vehicles have been spawned
        if len(self.initial_ids) < self.k.vehicle.num_vehicles:
            logging.error("Not enough vehicles have spawned! Bad start?")
            sys.exit()

        # save the initial state. This is used in the _reset function
        for veh_id in self.initial_ids:
            type_id = self.scenario.vehicles.get_type(veh_id)
            pos = self.k.vehicle.get_position(veh_id)
            lane = self.k.vehicle.get_lane(veh_id)
            speed = self.k.vehicle.get_speed(veh_id)
            route_id = "route" + self.k.vehicle.get_edge(veh_id)

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
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sumo_params.render:
                self.k.vehicle.update_vehicle_colors()

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
                if key in self.k.vehicle.get_arrived_ids():
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

        If "vehicle_arrangement_shuffle" is set to True in env_params, the
        vehicles swap initial positions with one another. Also, if a
        "starting_position_shuffle" is set to True, the initial position of
        vehicles are redone.

        If "warmup_steps" is set to a value greater than 0, then this method
        also runs the necessary number of warmup steps before beginning
        training, with actions to the agents being assigned by the simulator.

        Returns
        -------
        observation: numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # reset the time counter
        self.time_counter = 0

        # warn about not using restart_instance when using inflows
        if len(self.net_params.inflows.get()) > 0 and \
                not self.sumo_params.restart_instance:
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

        if self.sumo_params.restart_instance or self.step_counter > 2e6:
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sumo_params.seed = random.randint(0, 1e5)

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            # restart the sumo instance
            self.restart_sumo(self.sumo_params)

        # perform shuffling (if requested)
        if self.starting_position_shuffle or self.vehicle_arrangement_shuffle:
            if self.starting_position_shuffle:
                x0 = np.random.uniform(0, self.k.scenario.length())
            else:
                x0 = self.initial_config.x0

            veh_ids = deepcopy(self.initial_ids)
            if self.vehicle_arrangement_shuffle:
                random.shuffle(veh_ids)

            initial_positions, initial_lanes = \
                self.k.scenario.generate_starting_positions(
                    initial_config=self.initial_config,
                    num_vehicles=len(self.initial_ids), x0=x0)

            initial_state = dict()
            for i, veh_id in enumerate(veh_ids):
                route_id = "route" + initial_positions[i][0]

                # replace initial routes, lanes, positions, and speeds to
                # reflect new values
                list_initial_state = list(self.initial_state[veh_id])
                list_initial_state[1] = route_id
                list_initial_state[2] = initial_lanes[i]
                list_initial_state[3] = initial_positions[i][1]
                initial_state[veh_id] = tuple(list_initial_state)

            self.initial_state = deepcopy(initial_state)

        # clear all vehicles from the network and the vehicles class
        for veh_id in self.k.kernel_api.vehicle.getIDList():  # FIXME: hack
            try:
                self.k.vehicle.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.k.vehicle.get_ids()):
            try:
                self.k.vehicle.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))

        # reintroduce the initial vehicles to the network
        for veh_id in self.initial_ids:
            type_id, route_id, lane_index, pos, speed = \
                self.initial_state[veh_id]

            try:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    route_id=route_id,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)
            except (FatalTraCIError, TraCIException):
                # if a vehicle was not removed in the first attempt, remove it
                # now and then reintroduce it
                self.k.vehicle.remove(veh_id)
                self.k.vehicle.add(
                    veh_id=veh_id,
                    type_id=type_id,
                    route_id=route_id,
                    lane=lane_index,
                    pos=pos,
                    speed=speed)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # update the information in each kernel to match the current state
        self.k.update(reset=True)

        # update the colors of vehicles
        if self.sumo_params.render:
            self.k.vehicle.update_vehicle_colors()

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
        performing actions specified by sumo.

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
                self.k.vehicle.get_ids(),
                key=self.k.vehicle.get_absolute_position)
            return sorted_ids, None
        else:
            return self.k.vehicle.get_ids(), None

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
        self.k.close()

        # close pyglet renderer
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
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
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
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
        human_idlist = self.k.vehicle.get_human_ids()
        machine_idlist = self.k.vehicle.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.k.scenario.max_speed()
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.k.vehicle.get_orientation(id))
                machine_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.k.vehicle.get_orientation(id))
                human_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.k.vehicle.get_timestep(id),
                 self.k.vehicle.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.k.vehicle.get_orientation(id))
            machine_dynamics.append(
                self.k.vehicle.get_speed(id)/max_speed)

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
                orientation = self.k.vehicle.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.k.vehicle.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)
