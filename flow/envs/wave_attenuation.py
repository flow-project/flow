from flow.envs.base_env import Env
from flow.core import rewards
from flow.core import multi_agent_rewards
from flow.controllers.car_following_models import IDMController
from flow.core.params import InitialConfig, NetParams

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

import numpy as np
from numpy.random import normal
from scipy.optimize import fsolve

import pdb


class WaveAttenuationEnv(Env):
    """
    Fully functional environment. Takes in an *acceleration* as an action.
    Reward function is negative norm of the difference between the velocities of
    each vehicle, and the target velocity. State function is a vector of the
    velocities for each vehicle.
    """

    @property
    def action_space(self):
        """
        Actions are a set of accelerations from 0 to 15m/s
        :return:
        """
        return Box(low=-np.abs(self.env_params.max_decel),
                   high=self.env_params.max_accel,
                   shape=(self.vehicles.num_rl_vehicles, ))

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        speed = Box(low=0, high=np.inf, shape=(self.vehicles.num_vehicles,))
        pos = Box(low=0., high=np.inf, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, pos))

    def apply_rl_actions(self, rl_actions):
        """
        See parent class
        """
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]
        self.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        See parent class
        """
        vel = np.array([self.vehicles.get_speed(veh_id)
                        for veh_id in self.vehicles.get_ids()])

        if any(vel < -100) or kwargs["fail"]:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / self.v_eq_max

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 8  # 0.25
        rl_actions = np.array(rl_actions)
        # reward += eta * (3 - np.mean(np.abs(rl_actions)))
        accel_threshold = 0
        np.tanh(np.mean(np.abs(rl_actions)))
        if np.mean(np.abs(rl_actions)) > accel_threshold:
            reward += eta * (accel_threshold - np.mean(np.abs(rl_actions)))

        # # punish large rl headways
        # reward += - self.vehicles["rl_0"]["headway"] / 150.

        return float(reward)

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        target_vel = self.env_params.additional_params["target_velocity"]
        length = self.scenario.length
        scaled_vel = [self.vehicles.get_speed(veh_id) / target_vel
                      for veh_id in self.sorted_ids]
        scaled_headway = [(self.vehicles.get_headway(veh_id) % length) / length
                          for veh_id in self.sorted_ids]

        # for stabilizing the ring: place the rl car at index 0 to maintain
        # continuity between rollouts
        indx_rl = [ind for ind, veh_id in enumerate(self.sorted_ids)
                   if veh_id in self.vehicles.get_rl_ids()][0]
        indx_sorted_ids = np.mod(np.arange(len(self.sorted_ids)) + indx_rl,
                                 len(self.sorted_ids))

        return np.array([[scaled_vel[i], scaled_headway[i]]
                         for i in indx_sorted_ids])

    def _reset(self):
        """
        See parent class.
        The sumo instance is restart with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # update the scenario
        initial_config = InitialConfig(bunching=50, min_gap=0)
        additional_net_params = {"length": np.random.choice(np.arange(200, 350)),
                                 "lanes": 1, "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # TODO(nish): figure out what exactly this does (ask @cathywu?)
        # - shouldn't need to reinstantiate scenario, right? 
        self.scenario = self.env_params.additional_params["scenario_type"](
            self.scenario.name, self.scenario.generator_class,
            self.scenario.vehicles, net_params, initial_config)

        # solve for the velocity upper bound of the ring
        def v_eq_max_function(v):
            num_veh = self.vehicles.num_vehicles - 1
            # maximum gap in the presence of one rl vehicle
            s_eq_max = (self.scenario.length - self.vehicles.num_vehicles * 5) / num_veh

            v0 = 30
            s0 = 2
            T = 1
            gamma = 4

            error = s_eq_max - (s0 + v * T) * (1 - (v / v0) ** gamma) ** (- 1 / 2)

            return error

        v_guess = 4.
        self.v_eq_max = fsolve(v_eq_max_function, v_guess)[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params["length"])
        print("v_max:", self.v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_sumo(sumo_params=self.sumo_params,
                          sumo_binary=self.sumo_params.sumo_binary)

        # perform the generic reset function
        observation = super()._reset()

        # run the experiment for a few steps with the rl vehicle acting as a
        # human vehicle (before beginning the learning portion of the rollout)
        num_pre_steps = 750
        if num_pre_steps > 0:
            for i in range(num_pre_steps):
                observation = self.pre_step()

        # reset the timer to zero
        self.time_counter = 0

        return observation

    def pre_step(self):
        self.time_counter += 1

        # rl controllers are embedded with IDM Controllers to simulate human
        rl_ids = self.vehicles.get_rl_ids()
        # driving at first
        self.embedded_controller = dict.fromkeys(rl_ids)
        for veh_id in rl_ids:
            self.embedded_controller[veh_id] = IDMController(veh_id)

        # perform accelerations for rl vehicles moving like human vehicles
        accel = []
        for veh_id in rl_ids:
            action = self.embedded_controller[veh_id].get_action(self)
            accel.append(action)
        self.apply_acceleration(rl_ids, acc=accel)

        # perform acceleration actions for controlled human-driven vehicles
        if len(self.vehicles.get_controlled_ids()) > 0:
            accel = []
            for veh_id in self.vehicles.get_controlled_ids():
                accel_contr = self.vehicles.get_acc_controller(veh_id)
                action = accel_contr.get_action(self)
                accel.append(action)
            self.apply_acceleration(self.vehicles.get_controlled_ids(), accel)

        # perform (optionally) routing actions for all vehicle in the network,
        # including rl and sumo-controlled vehicles
        routing_ids = []
        routing_actions = []
        for veh_id in self.vehicles.get_ids():
            if self.vehicles.get_routing_controller(veh_id) is not None:
                routing_ids.append(veh_id)
                route_contr = self.vehicles.get_routing_controller(veh_id)
                routing_actions.append(route_contr.choose_route(self))

        self.choose_routes(veh_ids=routing_ids, route_choices=routing_actions)

        self.traci_connection.simulationStep()

        # collect information on the vehicle in the network from sumo
        vehicle_obs = self.traci_connection.vehicle.getSubscriptionResults()

        # get vehicle ids for the entering, exiting, and colliding vehicles
        id_lists = self.traci_connection.simulation.getSubscriptionResults()

        # store the network observations in the vehicles class
        self.vehicles.update(vehicle_obs, id_lists, self)

        # collect list of sorted vehicle ids
        self.sorted_ids, self.sorted_extra_data = self.sort_by_position()

        # collect information of the state of the network based on the
        # environment class used
        if isinstance(self.action_space, list):
            # rllab requires non-multi agent to have state shape as
            # num-states x num_vehicles
            self.state = self.get_state()
        else:
            self.state = self.get_state().T

        # collect observation new state associated with action
        next_observation = list(self.state)

        return next_observation


class WaveAttenuationPOEnv(WaveAttenuationEnv):
    """
    POMDP version of wave attenuation env
    """

    @property
    def observation_space(self):
        """
        See parent class
        An observation is an array the velocities for each vehicle
        """
        return Tuple((Box(low=-1, high=1, shape=(4,)),))
        # return Box(low=-1, high=1, shape=(4,))

    def get_state(self, **kwargs):
        """
        See parent class
        The state is an array the velocities for each vehicle
        :return: a matrix of velocities and absolute positions for each vehicle
        """
        vehID = self.vehicles.get_rl_ids()[0]
        lead_id = self.vehicles.get_leader(vehID)
        max_speed = 15.
        max_scenario_length = 350.

        if lead_id is None:
            lead_id = vehID
            self.vehicles.set_headway(vehID, 0)

        observation = np.array([
            [self.vehicles.get_speed(vehID) / max_speed],
            [(self.vehicles.get_speed(lead_id) - self.vehicles.get_speed(
                vehID)) / max_speed],
            [self.vehicles.get_headway(vehID) / max_scenario_length],
            [self.scenario.length / max_scenario_length]
        ])

        return observation
