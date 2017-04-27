import random
import math
from cistar.controllers.base_controller import BaseController
import collections

"""Contains a bunch of car-following control models for CISTAR.
Controllers can have their output delayed by some duration.
Each controller includes functions
    get_action(self, env) -> acc
        - using the current state of the world and existing parameters,
        uses the control model to return a vehicle acceleration.
    reset_delay(self) -> None
        - clears the queue of acceleration outputs used to generate
        delayed output. used when the experiment is reset to clear out 
        old actions based on old states.
"""

class CFMController(BaseController):
    """Basic car-following model. Only looks ahead.
    """

    def __init__(self, veh_id, k_d=1, k_v=1, k_c = 1, d_des=1, v_des = 8, acc_max = 20, tau = 0, dt = 0.1):
        """Instantiates a CFM controller
        
        Arguments:
            veh_id -- Vehicle ID for SUMO identification
        
        Keyword Arguments:
            k_d {number} -- [headway gain] (default: {1})
            k_v {number} -- [gain on difference between lead velocity and current] (default: {1})
            k_c {number} -- [gain on difference from desired velocity to current] (default: {1})
            d_des {number} -- [desired headway] (default: {1})
            v_des {number} -- [desired velocity] (default: {8})
            acc_max {number} -- [max acceleration] (default: {15})
            tau {number} -- [time delay] (default: {0})
            dt {number} -- [timestep] (default: {0.1})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": acc_max}
        BaseController.__init__(self, veh_id, controller_params)
        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des
        self.acc_max = acc_max
        self.accel_queue = collections.deque()

    def get_action(self, env):
        this_lane = env.vehicles[self.veh_id]['lane']

        lead_id = env.get_leading_car(self.veh_id, this_lane)
        if not lead_id: # no car ahead
            return self.acc_max

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']

        d_l = (lead_pos - this_pos) % env.scenario.length

        acc = self.k_d*(d_l - self.d_des) + self.k_v*(lead_vel - this_vel) + self.k_c*(self.v_des - this_vel)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb filling (currently), etc
            self.accel_queue.appendleft(acc)

        return min(self.accel_queue.pop(), self.acc_max)

    def reset_delay(self, env):
        self.accel_queue.clear()

class BCMController(BaseController):
    """Bilateral car-following model. Looks ahead and behind.
    
    [description]
    
    Variables:
    """

    def __init__(self, veh_id, k_d=1, k_v=1, k_c = 1, d_des=1, v_des = 8, acc_max = 15, deacc_max = -5, tau = 0, dt = 0.1):
        """Instantiates a BCM controller
        
        Arguments:
            veh_id -- Vehicle ID for SUMO identification
        
        Keyword Arguments:
            k_d {number} -- [gain on distances to lead/following cars] (default: {1})
            k_v {number} -- [gain on vehicle velocity differences] (default: {1})
            k_c {number} -- [gain on difference from desired velocity to current] (default: {1})
            d_des {number} -- [desired headway] (default: {1})
            v_des {number} -- [desired velocity] (default: {8})
            acc_max {number} -- [max acceleration] (default: {15})
            tau {number} -- [time delay] (default: {0})
            dt {number} -- [timestep] (default: {0.1})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": deacc_max}
        BaseController.__init__(self, veh_id, controller_params)
        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des
        self.acc_max = acc_max
        self.accel_queue = collections.deque()

    def get_action(self, env):
        # From the paper: 
        # There would also be additional control rules that take
        # into account minimum safe separation, relative speeds,
        # speed limits, weather and lighting conditions, traffic density
        # and traffic advisories

        this_lane = env.vehicles[self.veh_id]['lane']

        lead_id = env.get_leading_car(self.veh_id, this_lane)
        if not lead_id: # no car ahead
            return self.acc_max

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']

        trail_id = env.get_trailing_car(self.veh_id, this_lane)
        trail_pos = env.get_x_by_id(trail_id)
        trail_vel = env.vehicles[trail_id]['speed']

        headway = (lead_pos - this_pos) % env.scenario.length # d_l

        footway = (this_pos - trail_pos) % env.scenario.length # d_f

        acc = self.k_d * (headway - footway) + \
            self.k_v * ((lead_vel - this_vel) - (this_vel - trail_vel)) + \
            self.k_c * (self.v_des - this_vel)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb filling (currently), etc
            self.accel_queue.appendleft(acc)

        return min(self.accel_queue.pop(), self.acc_max)

    def reset_delay(self, env):
        self.accel_queue.clear()

class OVMController(BaseController):
    """Optimal Vehicle Model, per Gabor
    
    [description]
    
    Variables:
    """

    def __init__(self, veh_id, alpha = 1, beta = 1, h_st = 2, h_go = 15, v_max = 30, acc_max = 15, max_deaccel=5, tau = 0, dt = 0.1):
        """Instantiates an OVM controller
        
         Arguments:
            veh_id -- Vehicle ID for SUMO identification
        
        Keyword Arguments:
            alpha {number} -- [gain on desired velocity to current velocity difference] (default: {0.6})
            beta {number} -- [gain on lead car velocity and self velocity difference] (default: {0.9})
            h_st {number} -- [headway for stopping] (default: {5})
            h_go {number} -- [headway for full speed] (default: {35})
            v_max {number} -- [max velocity] (default: {30})
            acc_max {number} -- [max acceleration] (default: {15})
            deacc_max {number} -- [max deceleration] (default: {-5})
            tau {number} -- [time delay] (default: {0.4})
            dt {number} -- [timestep] (default: {0.1})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": max_deaccel}
        BaseController.__init__(self, veh_id, controller_params)
        self.accel_queue = collections.deque()
        self.max_deaccel = max_deaccel
        self.acc_max = acc_max
        self.veh_id = veh_id
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta
        self.h_st = h_st
        self.h_go = h_go
        self.tau = tau
        self.dt = dt
        
    def get_action(self, env):
        this_lane = env.vehicles[self.veh_id]['lane']

        lead_id = env.get_leading_car(self.veh_id, this_lane)
        if not lead_id: # no car ahead
            return self.acc_max

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']

        h = (lead_pos - lead_length - this_pos) % env.scenario.length
        h_dot = lead_vel - this_vel

        # V function here - input: h, output : Vh
        if h <= self.h_st:
            Vh = 0
        elif self.h_st < h < self.h_go:
            Vh = self.v_max / 2 * (1 - math.cos(math.pi * (h - self.h_st) / (self.h_go - self.h_st)))
        else:
            Vh = self.v_max

        acc = self.alpha*(Vh - this_vel) + self.beta*(h_dot)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb filling (currently), etc
            self.accel_queue.appendleft(acc)

        return max(min(self.accel_queue.pop(), self.acc_max), -1 * self.max_deaccel)

    def reset_delay(self, env):
        self.accel_queue.clear()

class LinearOVM(BaseController):
    """Optimal Vehicle Model, 
    sources: 
        "Metastability in the formation of an experimental traffic jam", Nakayama
        _Traffic Flow Dynamics_ (a textbook), Treiber & Kesting
            like pg 170
            includes a caveat about this OVM - is this a deal-breaker for use in experiments?
            
    
    [description]
    
    Variables:
    """

    def __init__(self, veh_id, v_max = 30, acc_max = 15, max_deaccel=5, adaptation = 0.65, delay_time = 0, dt = 0.1):
        """Instantiates an OVM controller
        
         Arguments:
            veh_id -- Vehicle ID for SUMO identification
        
        Keyword Arguments:
            alpha {number} -- [gain on desired velocity to current velocity difference] (default: {0.6})
            beta {number} -- [gain on lead car velocity and self velocity difference] (default: {0.9})
            h_st {number} -- [headway for stopping] (default: {5})
            h_go {number} -- [headway for full speed] (default: {35})
            v_max {number} -- [max velocity] (default: {30})
            acc_max {number} -- [max acceleration] (default: {15})
            deacc_max {number} -- [max deceleration] (default: {-5})
            tau {number} -- [time delay] (default: {0.4})
            dt {number} -- [timestep] (default: {0.1})
        """

        controller_params = {"delay": delay_time/dt, "max_deaccel": max_deaccel}
        BaseController.__init__(self, veh_id, controller_params)
        self.accel_queue = collections.deque()
        self.max_deaccel = max_deaccel
        self.acc_max = acc_max
        self.veh_id = veh_id
        self.v_max = v_max # 4.8*1.85 for case I, 3.8*1.85 for case II, per Nakayama
        self.adaptation = adaptation # TAU in Traffic Flow Dynamics textbook
        self.delay_time = delay_time
        self.dt = dt

        
    def get_action(self, env):
        this_lane = env.vehicles[self.veh_id]['lane']

        lead_id = env.get_leading_car(self.veh_id, this_lane)
        if not lead_id: # no car ahead
            return self.acc_max

        lead_pos = env.get_x_by_id(lead_id)
        lead_vel = env.vehicles[lead_id]['speed']
        lead_length = env.vehicles[lead_id]['length']

        this_pos = env.get_x_by_id(self.veh_id)
        this_vel = env.vehicles[self.veh_id]['speed']

        h = (lead_pos - lead_length - this_pos) % env.scenario.length
        h_dot = lead_vel - this_vel

        # V function here - input: h, output : Vh
        alpha = 1.689 # the average value from Nakayama paper
        if h < 5:
            Vh = 0
        elif 5 <= h <= 5 + self.v_max/alpha:
            Vh = alpha * (h - 5)
        else:
            Vh = self.v_max


        acc = (Vh - this_vel) / self.adaptation

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb filling (currently), etc
            self.accel_queue.appendleft(acc)

        return max(min(self.accel_queue.pop(), self.acc_max), -1 * self.max_deaccel)

    def reset_delay(self, env):
        self.accel_queue.clear()

