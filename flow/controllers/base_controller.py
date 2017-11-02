"""
This file contains the base controllers used by human-driven vehicle units.

Two types of controllers are provided:
 - BaseController: A controller that instantiates a vehicle with car-following
   dynamics controlled by acceleration models in flow_dev (located in
   car_following_models.py)
 - SumoController: A controller that instantiates a vehicle with car-following
   dynamics from sumo's built-in functions
"""

import numpy as np
import collections


class BaseController:

    def __init__(self, veh_id, controller_params):
        """
        Base class for flow-controlled acceleration behavior.

        Instantiates a controller and forces the user to pass a
        maximum acceleration to the controller. Provides the method
        safe_action to ensure that controls are never made that could
        cause the system to crash.

        Attributes
        ----------
        veh_id: string
            ID of the vehicle this controller is used for
        controller_params: dict
            Dictionary that optionally contains 'delay', the delay, and must
            contain 'max_deaccel', the maximum deceleration as well as all
            other parameters that dictate the driving behavior.
        """
        self.d = 0
        self.veh_id = veh_id
        self.controller_params = controller_params

        # magnitude of gaussian noise
        if "noise" not in controller_params:
            self.acc_noise = 0
        else:
            self.acc_noise = controller_params["noise"]

        if not controller_params['delay']:
            self.delay = 0
        else:
            self.delay = controller_params['delay']

        # max deaccel should always be a positive
        self.max_deaccel = np.abs(controller_params['max_deaccel'])
        self.acc_queue = collections.deque()

    def reset_delay(self, env):
        raise NotImplementedError

    def get_accel(self, env):
        """
        Returns the acceleration of the controller
        """
        raise NotImplementedError

    def get_action(self, env):
        """
        Returns the acceleration requested by get_accel pull some stochastic
        noise (if requested by the controller).
        """
        accel = self.get_accel(env)

        if self.acc_noise > 0:
            accel += np.random.normal(0, self.acc_noise)

        return accel

    def get_safe_action_instantaneous(self, env, action):
        """
        Instantaneously stops the car if there is a change of colliding into
        the leading vehicle in the next step

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step
        action: float
            requested acceleration action

        Returns
        -------
        safe_action: float
            the requested action if it does not lead to a crash; and a stopping
            action otherwise
        """
        # if there is only one vehicle in the network, all actions are safe
        if env.vehicles.num_vehicles == 1:
            return action

        lead_id = env.vehicles.get_leader(self.veh_id)

        # if there is no other vehicle in the current lane, all actions are safe
        if lead_id is None:
            return action

        this_vel = env.vehicles.get_speed(self.veh_id)
        time_step = env.time_step
        next_vel = this_vel + action * time_step
        h = env.vehicles.get_headway(self.veh_id)

        if next_vel > 0:
            if h < time_step * next_vel + this_vel * 1e-3:
                # if the vehicle will crash into the vehicle ahead of it in the
                # next time step (assuming the vehicle ahead of it is not
                # moving), then stop immediately
                return -this_vel / time_step
            else:
                # if the vehicle is not in danger of crashing, continue with the
                # requested action
                return action
        else:
            return action

    def get_safe_action(self, env, action):
        """
        Checks if the computed acceleration would put us above safe velocity.
        If it would, output the acceleration that would put at to safe velocity.

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step
        action: float
            requested acceleration action

        Returns
        -------
        safe_action: float
            the requested action clipped by the safe velocity
        """
        if env.vehicles.num_vehicles == 1:
            # if there is only one vehicle in the network, all actions are safe
            return action
        else:
            safe_velocity = self.safe_velocity(env)

            this_vel = env.vehicles.get_speed(self.veh_id)
            time_step = env.time_step

            if this_vel + action * time_step > safe_velocity:
                return (safe_velocity - this_vel)/time_step
            else:
                return action

    def safe_velocity(self, env):
        """
        Finds maximum velocity such that if the lead vehicle were to stop
        entirely, we can bring the following vehicle to rest at the point at
        which the headway is zero.

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step

        Returns
        -------
        safe_velocity: float
            maximum safe velocity given a maximum deceleration and delay in
            performing the breaking action
        """
        lead_id = env.vehicles.get_leader(self.veh_id)
        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)

        h = env.vehicles.get_headway(self.veh_id)
        dv = lead_vel - this_vel

        v_safe = 2 * h / env.time_step + dv - this_vel * (2 * self.delay)

        return v_safe


# TODO: still a work in progress
class SumoController:

    def __init__(self, veh_id, controller_params):
        """
        Base class for sumo-controlled acceleration behavior.

        Attributes
        ----------
        veh_id: str
            unique vehicle identifier
        controller_params: dict, optional
            contains the parameters needed to instantiate a sumo controller
            - model_type {string} -- type of SUMO car-following model to use.
              Must be one of: Krauss, KraussOrig1, PWagner2009, BKerner, IDM,
              IDMM, KraussPS, KraussAB, SmartSK, Wiedemann, Daniel1
            - model_params {dict} -- dictionary of parameters applicable to sumo
              cars, see: http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
        """
        self.veh_id = veh_id

        available_models = ["Krauss", "KraussOrig1", "PWagner2009", "BKerner",
                            "IDM", "IDMM", "KraussPS", "KraussAB", "SmartSK",
                            "Wiedemann", "Daniel1"]

        if "model_type" in controller_params:
            # the model type specified must be available in sumo
            if controller_params["model_type"] not in available_models:
                raise ValueError("Model type is not available in SUMO.")

            self.model_type = controller_params["model"]
        else:
            # if no model is specified, the controller defaults to sumo's
            # Krauss model
            self.model_type = "Krauss"

        if "model_params" in controller_params:
            self.model_params = controller_params["model_params"]
        else:
            self.model_params = dict()
