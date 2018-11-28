from copy import deepcopy
import numpy as np
import random
import traceback
from gym.spaces import Box

from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException

from ray.rllib.env import MultiAgentEnv

from flow.envs.base_env import Env


class MultiEnv(MultiAgentEnv, Env):
    """Multi-agent version of base env. See parent class for info"""

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicles and TrafficLights classes, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions: numpy ndarray
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation: dict of numpy ndarrays
            agent's observation of the current environment
        reward: dict of floats
            amount of reward associated with the previous state/action pair
        done: dict of bools
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

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
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

            self.traci_connection.simulationStep()

            # collect subscription information from sumo
            vehicle_obs = \
                self.traci_connection.vehicle.getSubscriptionResults()
            id_lists = \
                self.traci_connection.simulation.getSubscriptionResults()
            tls_obs = \
                self.traci_connection.trafficlight.getSubscriptionResults()

            # store new observations in the vehicles and traffic lights class
            self.vehicles.update(vehicle_obs, id_lists, self)
            self.traffic_lights.update(tls_obs)

            # update the colors of vehicles
            self.update_vehicle_colors()

            # collect list of sorted vehicle ids
            self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

            # crash encodes whether the simulator experienced a collision
            crash = \
                self.traci_connection.simulation.getStartingTeleportNumber() \
                != 0

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

        states = self.get_state()
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
            # test if the agent has exited the system
            if key in self.vehicles.get_arrived_ids():
                done[key] = True
            # check if an agent is done
            if crash:
                done['__all__'] = True
            else:
                done['__all__'] = False
            infos[key] = {}

        reward = self.compute_reward(rl_actions, fail=crash)

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
        observation: dict of numpy ndarrays
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        # reset the time counter
        self.time_counter = 0

        # warn about not using restart_instance when using inflows
        if len(self.scenario.net_params.inflows.get()) > 0 and \
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
            # modify the vehicles class to match initial data
            self.vehicles = deepcopy(self.initial_vehicles)
            # restart the sumo instance
            self.restart_sumo(self.sumo_params)

        # perform shuffling (if requested)
        if self.starting_position_shuffle or self.vehicle_arrangement_shuffle:
            if self.starting_position_shuffle:
                x0 = np.random.uniform(0, self.scenario.length)
            else:
                x0 = self.scenario.initial_config.x0

            veh_ids = deepcopy(self.initial_ids)
            if self.vehicle_arrangement_shuffle:
                random.shuffle(veh_ids)

            initial_positions, initial_lanes = \
                self.scenario.generate_starting_positions(
                    num_vehicles=len(self.initial_ids), x0=x0)

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

            self.initial_state = deepcopy(initial_state)

        # clear all vehicles from the network and the vehicles class
        for veh_id in self.traci_connection.vehicle.getIDList():
            try:
                self.traci_connection.vehicle.remove(veh_id)
                self.traci_connection.vehicle.unsubscribe(veh_id)
                self.vehicles.remove(veh_id)
            except (FatalTraCIError, TraCIException):
                print("Error during start: {}".format(traceback.format_exc()))
                pass

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.vehicles.get_ids()):
            self.vehicles.remove(veh_id)
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

        self.traci_connection.simulationStep()

        # collect subscription information from sumo
        vehicle_obs = self.traci_connection.vehicle.getSubscriptionResults()
        id_lists = self.traci_connection.simulation.getSubscriptionResults()
        tls_obs = self.traci_connection.trafficlight.getSubscriptionResults()

        # store new observations in the vehicles and traffic lights class
        self.vehicles.update(vehicle_obs, id_lists, self)
        self.traffic_lights.update(tls_obs)

        # update the colors of vehicles
        self.update_vehicle_colors()

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
        self.state = {}
        observation = {}
        for key, state in states.items():
            # collect information of the state of the network based on the
            # environment class used
            self.state[key] = np.asarray(state).T

            # collect observation new state associated with action
            observation[key] = np.copy(self.state[key]).tolist()

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)

        return observation

    def apply_rl_actions(self, rl_actions=None):
        """Specify the actions to be performed by the rl agent(s).

        If no actions are provided at any given step, the rl agents default to
        performing actions specified by sumo.

        Parameters
        ----------
        rl_actions: list or numpy ndarray
            list of actions provided by the RL algorithm
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        # clip according to the action space requirements
        if isinstance(self.action_space, Box):
            for key, action in rl_actions.items():
                rl_actions[key] = np.clip(
                    action,
                    a_min=self.action_space.low,
                    a_max=self.action_space.high)

        self._apply_rl_actions(rl_actions)
