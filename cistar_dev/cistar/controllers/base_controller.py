"""
This file contains the base controllers used by human-driven vehicle units.

Two types of controllers are provided:
 - BaseController: A controller that instantiates a vehicle with car-following
   dynamics controlled by acceleration models in cistar_dev (located in
   car_following_models.py)
 - SumoController: A controller that instantiates a vehicle with car-following
   dynamics from sumo's built-in functions
"""

import numpy as np
import collections

class BaseController:
    """
    Base class for cistar-controlled acceleration behavior.

    Instantiates a controller and forces the user to pass a
    maximum acceleration to the controller. Provides the method
    safe_action to ensure that controls are never made that could
    cause the system to crash.
    """

    def __init__(self, veh_id, controller_params):
        """
        Arguments
        ---------
        veh_id: string
            ID of the vehicle this controller is used for
        controller_params: dict
            Dictionary that optionally contains 'delay', the delay, and must
            contain 'max_deaccel', the maximum deacceleration as well as all
            other parameters that dictate the driving behavior.
        """
        self.d = 0
        self.veh_id = veh_id
        self.controller_params = controller_params

        if not controller_params['delay']:
            self.delay = 0
        else:
            self.delay = controller_params['delay']
        # max deaccel should always be a positive
        self.max_deaccel = np.abs(controller_params['max_deaccel'])
        self.acc_queue = collections.deque()

    def reset_delay(self, env):
        raise NotImplementedError

    def get_action(self, env):
        """
        Returns the acceleration of the controller
        """
        raise NotImplementedError

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
        # if there is only one vehicle in the environment, all actions are safe
        if len(env.vehicles) == 1:
            return action

        lead_id = env.vehicles[self.veh_id]["leader"]

        # if there is no other vehicle in the current lane, all actions are safe
        if lead_id is None:
            return action

        this_vel = env.vehicles[self.veh_id]['speed']
        time_step = env.time_step
        next_vel = this_vel + action * time_step
        h = env.vehicles[self.veh_id]["headway"]

        if next_vel > 0:
            if h < time_step * next_vel + this_vel * 1e-3:
                return -this_vel / time_step
            else:
                return action
        else:
            return action

    def get_safe_action(self, env, action):
        """
        USE THIS INSTEAD OF GET_ACTION for computing the actual controls.
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

        if len(env.vehicles) == 1:
            return action
        else:
            safe_velocity = self.safe_velocity(env)

            this_vel = env.vehicles[self.veh_id]['speed']
            time_step = env.time_step

            if this_vel + action * time_step > safe_velocity:
                return (safe_velocity - this_vel)/time_step
            else:
                return action

    def safe_velocity(self, env):
        """
        Finds maximum velocity such that if the lead vehicle breaks
        with max deceleration, we can bring the following vehicle to rest
        at the point at which the headway is zero.

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
        this_lane = env.vehicles[self.veh_id]['lane']
        lead_id = env.vehicles[self.veh_id]["leader"]

        lead_pos = env.vehicles[lead_id]["absolute_position"]
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.vehicles[self.veh_id]["absolute_position"]

        # need to account for the position being reset around the length
        self.max_deaccel = np.abs(self.max_deaccel)
        if lead_pos > this_pos:
            dist = lead_pos - (this_pos + lead_length)
        else:
            loop_length = env.scenario.net_params["length"]
            dist = (this_pos + lead_length) - (lead_pos + loop_length)

        d = dist - np.power((lead_vel - self.max_deaccel * env.time_step), 2)/(2*self.max_deaccel)

        if -2*d+self.max_deaccel*self.delay**2 < 0:
            v_safe = 0
        else:
            v_safe = (-self.max_deaccel*self.delay +
                np.sqrt(self.max_deaccel)*np.sqrt(-2*d+self.max_deaccel*self.delay**2))

        return v_safe


# TODO: still a work in progress
class SumoController:
    """
    Base class for sumo-controlled acceleration behavior.
    """

    def __init__(self, veh_id, controller_params):
        """
        Initializes a SUMO controller with information required by sumo.

        :param veh_id {string} -- unique vehicle identifier
        :param controller_params {dict} -- contains the parameters needed to instantiate a sumo controller
               - model_type {string} -- type of SUMO car-following model to use. Must be one of: Krauss, KraussOrig1,
                 PWagner2009, BKerner, IDM, IDMM, KraussPS, KraussAB, SmartSK, Wiedemann, Daniel1
               - model_params {dict} -- dictionary of parameters applicable to sumo cars,
                 see: http://sumo.dlr.de/wiki/Definition_of_Vehicles,_Vehicle_Types,_and_Routes
        """
        self.veh_id = veh_id

        available_models = ["Krauss", "KraussOrig1", "PWagner2009", "BKerner", "IDM", "IDMM", "KraussPS",
                            "KraussAB", "SmartSK", "Wiedemann", "Daniel1"]

        if "model_type" in controller_params:
            # the model type specified must be available in sumo
            if controller_params["model_type"] not in available_models:
                raise ValueError("Model type is not available in SUMO.")

            self.model_type = controller_params["model"]
        else:
            # if no model is specified, the controller defaults to sumo's Krauss model
            self.model_type = "Krauss"

        if "model_params" in controller_params:
            self.model_params = controller_params["model_params"]
        else:
            self.model_params = dict()
