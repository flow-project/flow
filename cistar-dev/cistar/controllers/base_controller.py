"""
This file contains the base controllers used by human-driven vehicle units.

Two types of controllers are provided:
 - BaseController: A controller that instantiates a vehicle with car-following dynamics
   controlled by acceleration models in cistar (located in car_following_models.py)
 - SumoController: A controller that instantiates a vehicle with car-following dynamics
   from sumo's built-in functions
"""

import numpy as np
import collections
import pdb


class BaseController:
    """ Base class for cistar-controlled acceleration behavior.

    Instantiates a controller and forces the user to pass a
    maximum acceleration to the controller. Provides the method
    safe_action to ensure that controls are never made that could
    cause the system to crash.

    """

    def __init__(self, veh_id, controller_params):
        """
        Arguments:
            veh_id {string} -- ID of the vehicle this controller is used for
            controller_params {Dictionary} -- Dictionary that optionally 
            contains 'delay', the delay, and must contain 
            'max_deaccel', the maximum deacceleration as well as all 
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
        """ Returns the acceleration of the controller """
        raise NotImplementedError

    def get_safe_action_instantaneous(self, env, action):
        """
        Instantaneously stops the car if there is a change of colliding into the leading vehicle in the next step
        :param env:
        :param action:
        :return:
        """
        # if there is only one vehicle in the environment, all actions are safe
        if len(env.vehicles) == 1:
            return action

        this_lane = env.vehicles[self.veh_id]['lane']
        lead_id = env.get_leading_car(self.veh_id, this_lane)

        # if there is no other vehicle in the current lane, all actions are safe
        if lead_id is None:
            return action

        lead_pos = env.get_x_by_id(lead_id)
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']
        time_step = env.time_step
        next_vel = this_vel + action * time_step

        h = (lead_pos - lead_length - this_pos) % env.scenario.length
        if next_vel > 0:
            if h < time_step * next_vel + this_vel * 1e-3:
                return -this_vel / time_step
            else:
                return action
        else:
            return action

    def get_safe_action(self, env, action):
        """ USE THIS INSTEAD OF GET_ACTION for computing the actual controls.
        Checks if the computed acceleration would put us above safe velocity.
        If it would, output the acceleration that would put at to safe velocity. 
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
        """
        this_lane = env.vehicles[self.veh_id]['lane']
        lead_id = env.get_leading_car(self.veh_id, this_lane)

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.get_x_by_id(self.veh_id)

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

    def get_safe_intersection_action(self, env, action):
        """ Fail-safe used to ensure vehicles do not collide at an intersection.

        Orders vehicles to stop if there are about to enter an intersection currently
        occupied by a vehicle moving perpendicular.

        Provides right-of-way to one side ("top-bottom" or "left-right") in case vehicles
        at the same time to an intersection.
        """
        time_step = env.time_step
        this_vel = env.vehicles[self.veh_id]['speed']
        next_vel = this_vel + action * time_step
        this_dist_to_intersection, this_intersection = env.get_distance_to_intersection(self.veh_id)

        stop_action = - this_vel / time_step

        # if the car is not about to enter the intersection, continue moving as requested

        if next_vel * time_step + this_vel * 1e-3 < this_dist_to_intersection:
            return action

        # if the vehicle is about to enter an intersection, and another vehicle is currently in the intersection
        # from the perpendicular end, stop

        # TODO: modify this for multiple intersections
        if env.intersection_edges[0] in this_intersection:
            cross_intersection = env.intersection_edges[1]
        elif env.intersection_edges[1] in this_intersection:
            cross_intersection = env.intersection_edges[0]

        # TODO: also make sure that the car is more than its vehicle length out of the intersection
        if any([cross_intersection in env.vehicles[veh_id]["edge"] for veh_id in env.ids]):
            return stop_action

        # if two cars are about to enter the intersection, and the other car has right of way, stop;
        # else, continue into the intersection

        other_dist = []
        other_veh_id = []
        for veh_id in env.ids:
            dist, intersection = env.get_distance_to_intersection(veh_id)

            if cross_intersection in intersection:
                other_dist.append(dist)
                other_veh_id.append(veh_id)

        # minimum distance from the cross end until a vehicle
        ind_min_cross_dist = np.argmin(other_dist)
        cross_dist = other_dist[ind_min_cross_dist]
        cross_veh_id = other_veh_id[ind_min_cross_dist]

        cross_vel = env.vehicles[cross_veh_id]["speed"]
        cross_max_vel = cross_vel + env.env_params["max-acc"] * time_step

        # TODO: this does not take into consideration what the velocity of the other vehicle may be in the next step
        # TODO: a possible move could be to add the maximum acceleration to the vehicle (worst case scenario)
        if cross_max_vel * time_step > cross_dist:
            # if this vehicle does not have right-of-way, stop
            if env.intersection_edges[1] in this_intersection and env.intersection_fail_safe == "left-right":
                return stop_action
            # if this vehicle does have right-of-way, continue
            elif env.intersection_edges[0] in this_intersection and env.intersection_fail_safe == "top-bottom":
                return action
        else:
            return action


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
