import numpy as np
import collections

"""Base class for controllers. 

Instantiates a controller and forces the user to pass a 
maximum acceleration to the controller. Provides the method
safe_action to ensure that controls are never made that could
cause the system to crash. 

"""
class BaseController:
    def __init__(self, veh_id, controller_params):
        """
        Arguments:
            veh_id {string} -- ID of the vehicle this controller is used for
            controller_params {Dictionary} -- Dictionary that optionally 
            contains 'delay', the delay, and must contain 
            'max_deaccel', the maximum deacceleration as well as all 
            other parameters that dictate the driving behavior. 
        """
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

    def safe_action(self, env, action):
        """ USE THIS INSTEAD OF GET_ACTION for computing the actual controls.
        Checks if the computed acceleration would put us above safe velocity.
        If it would, output the acceleration that would put at to safe velocity. 
        """
        safe_velocity = self.safe_velocity(env)

        #this is not being used?
        this_lane = env.vehicles[self.veh_id]['lane']

        this_vel = env.vehicles[self.veh_id]['speed']
        time_step = env.time_step

        if this_vel + action*time_step > safe_velocity:
            return (safe_velocity - this_vel)/time_step
        else:
            return action


    def safe_velocity(self, env):
        """Finds maximum velocity such that if the lead vehicle breaks
        with max acceleration, we can bring the following vehicle to rest
        at the point at which the headway is zero.
        """
        this_lane = env.vehicles[self.veh_id]['lane']
        lead_id = env.get_leading_car(self.veh_id, this_lane)

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.get_x_by_id(self.veh_id)

        # need to account for the position being reset around the length
        if lead_pos > this_pos: 
            d = (this_pos + lead_length) - lead_pos - np.power((lead_vel),2)/(2*self.max_deaccel)
        else:
            loop_length = env.scenario.net_params["length"]
            d = (this_pos + lead_length) - (lead_pos + loop_length) - np.power((lead_vel),2)/(2*self.max_deaccel)

        v_safe = (-self.max_deaccel*self.delay +
                np.sqrt(self.max_deaccel)*np.sqrt(-2*d+self.max_deaccel*self.delay**2))

        return v_safe