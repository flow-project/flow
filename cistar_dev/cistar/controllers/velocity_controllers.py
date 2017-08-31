from cistar.controllers.base_controller import BaseController
import numpy as np

class ConstantVelocityController(BaseController):
    """Base velocity controller (assumes acceleration by Default)
    """

    def __init__(self, veh_id, max_deaccel=15, max_accel = 6, tau=0, dt=0.1, constant_speed=7):
        """Instantiates a velocity controller

        Arguments:
            veh_id -- Vehicle ID for SUMO identification

        Keyword Arguments:
            max_deaccel {number} -- [max deacceleration] (default: {15})
            tau {number} -- [time delay] (default: {0})
            dt {number} -- [timestep] (default: {0.1})
            constant_speed {number} -- [target constant velocity] (default: {15})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": max_deaccel}
        BaseController.__init__(self, veh_id, controller_params)
        self.constant_speed = constant_speed
        self.max_deaccel = -abs(max_deaccel)
        self.max_accel = max_accel

    def get_action(self, env):
        this_vel = env.vehicles[self.veh_id]['speed']
        acc = (self.constant_speed - this_vel)/env.time_step
        if acc > 0:
            if acc > self.max_accel:
                return self.max_accel
            else:
                return acc
        else:
            if acc > self.max_deaccel:
                return acc
            else:
                return self.max_deaccel

    def get_safe_action(self, env, action):
        v_safe = self.safe_velocity(env)
        if v_safe < action:
            print(v_safe, action)
        return min(action, v_safe)

    def reset_delay(self, env):
        pass


class FollowerStopper(BaseController):
    """Inspired by Dan Work's... work.
    
    [description]
    
    Extends:
        BaseController
    """

    def __init__(self, veh_id, max_deaccel=15, tau=0, dt=0.1, constant_speed=15):
        controller_params = {"delay": tau/dt, "max_deaccel": max_deaccel}
        BaseController.__init__(self, veh_id, controller_params)
        self.constant_speed = constant_speed

    def get_action(self, env):
        this_lane = env.vehicles[self.veh_id]['lane']

        lead_id = env.vehicles[self.veh_id]['leader']
        if not lead_id: # no car ahead
            return self.acc_max

        lead_pos = env.vehicles[lead_id]['absolute_position']
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.vehicles[self.veh_id]['absolute_position']
        this_vel = env.vehicles[self.veh_id]['speed']

        deltaV = lead_vel - this_vel
        deltaX0 = np.array([4.5, 5.25, 6])  # initial values
        d_j = np.array([1.5, 1, .5])  # decel rates

        deltaX = deltaX0 + 1/(2*d_j) * (min(deltaV, 0))**2  # A function of deltaV

        dx = (lead_pos - lead_length - this_pos) % env.scenario.length  # headway

        this_v = min(max(lead_vel, 0), self.constant_speed)
        if dx < deltaX[0]:
            v_cmd = 0
        elif dx < deltaX[1]:
            v_cmd = this_v*(dx - deltaX[0])/(deltaX[1] - deltaX[0])
        elif dx < deltaX[2]:
            v_cmd = this_v + (self.constant_speed - this_v)*(dx - deltaX[1])/(deltaX[2] - deltaX[1])
        else:
            v_cmd = self.constant_speed
        return v_cmd

    def get_safe_action(self, env, action):
        v_safe = self.safe_velocity(env)
        if v_safe < action:
            print(v_safe, action)
        return min(action, v_safe)
