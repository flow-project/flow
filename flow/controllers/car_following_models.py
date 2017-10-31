"""
This script contains several car-following control models for flow-controlled
vehicles.

Controllers can have their output delayed by some duration. Each controller
includes functions
    get_accel(self, env) -> acc
        - using the current state of the world and existing parameters,
        uses the control model to return a vehicle acceleration.
    reset_delay(self) -> None
        - clears the queue of acceleration outputs used to generate
        delayed output. used when the experiment is reset to clear out
        old actions based on old states.
"""

import random
import math
from flow.controllers.base_controller import BaseController
import collections
import numpy as np


class CFMController(BaseController):

    def __init__(self, veh_id, k_d=1, k_v=1, k_c=1, d_des=1, v_des=8,
                 accel_max=20, decel_max=-5, tau=0.5, dt=0.1, noise=0):
        """
        Instantiates a CFM controller
        
        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        k_d: float
            headway gain (default: 1)
        k_v: float, optional
            gain on difference between lead velocity and current (default: 1)
        k_c: float, optional
            gain on difference from desired velocity to current (default: 1)
        d_des: float, optional
            desired headway (default: 1)
        v_des: float, optional
            desired velocity (default: 8)
        accel_max: float
            max acceleration (default: 20)
        decel_max: float
            max deceleration (default: -5)
        tau: float, optional
            time delay (default: 0)
        dt: float, optional
            timestep (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        """
        controller_params = {"delay": tau/dt, "max_deaccel": decel_max,
                             "noise": noise}
        BaseController.__init__(self, veh_id, controller_params)
        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des
        self.accel_max = accel_max
        self.accel_queue = collections.deque()

    def get_accel(self, env):
        lead_id = env.vehicles.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.accel_max

        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)

        d_l = env.vehicles.get_headway(self.veh_id)

        acc = self.k_d*(d_l - self.d_des) + self.k_v*(lead_vel - this_vel) + \
            self.k_c*(self.v_des - this_vel)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb
            # filling (currently), etc
            self.accel_queue.appendleft(acc)

        return min(self.accel_queue.pop(), self.accel_max)

    def reset_delay(self, env):
        self.accel_queue.clear()


class BCMController(BaseController):

    def __init__(self, veh_id, k_d=1, k_v=1, k_c=1, d_des=1, v_des=8,
                 accel_max=15, decel_max=-5, tau=0.5, dt=0.1, noise=0):
        """
        Instantiates a Bilateral car-following model controller. Looks ahead
        and behind.
        
        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        k_d: float, optional
            gain on distances to lead/following cars (default: 1)
        k_v: float, optional
            gain on vehicle velocity differences (default: 1)
        k_c: float, optional
            gain on difference from desired velocity to current (default: 1)
        d_des: float, optional
            desired headway (default: 1)
        v_des: float, optional
            desired velocity (default: 8)
        accel_max: float, optional
            max acceleration (default: 15)
        decel_max: float
            max deceleration (default: -5)
        tau: float, optional
            time delay (default: 0.5)
        dt: float, optional
            timestep (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        """
        controller_params = {"delay": tau / dt, "max_deaccel": decel_max,
                             "noise": noise}
        BaseController.__init__(self, veh_id, controller_params)
        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des
        self.accel_max = accel_max
        self.accel_queue = collections.deque()

    def get_accel(self, env):
        """
        From the paper:
        There would also be additional control rules that take
        into account minimum safe separation, relative speeds,
        speed limits, weather and lighting conditions, traffic density
        and traffic advisories
        """
        lead_id = env.vehicles.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.accel_max

        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)

        trail_id = env.vehicles.get_follower(self.veh_id)
        trail_vel = env.vehicles.get_speed(trail_id)

        headway = env.vehicles.get_headway(self.veh_id)
        footway = env.vehicles.get_headway(trail_id)

        acc = self.k_d * (headway - footway) + \
            self.k_v * ((lead_vel - this_vel) - (this_vel - trail_vel)) + \
            self.k_c * (self.v_des - this_vel)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb
            # filling (currently), etc
            self.accel_queue.appendleft(acc)

        return min(self.accel_queue.pop(), self.accel_max)

    def reset_delay(self, env):
        self.accel_queue.clear()


class OVMController(BaseController):

    def __init__(self, veh_id, alpha=1, beta=1, h_st=2, h_go=15, v_max=30,
                 accel_max=15, decel_max=-5, tau=0.5, dt=0.1, noise=0):
        """
        Instantiates an Optimal Vehicle Model controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        alpha: float, optional
            gain on desired velocity to current velocity difference
            (default: 0.6)
        beta: float, optional
            gain on lead car velocity and self velocity difference
            (default: 0.9)
        h_st: float, optional
            headway for stopping (default: 5)
        h_go: float, optional
            headway for full speed (default: 35)
        v_max: float, optional
            max velocity (default: 30)
        accel_max: float, optional
            max acceleration (default: 15)
        decel_max: float, optional
            max deceleration (default: -5)
        tau: float, optional
            time delay (default: 0.5)
        dt: float, optional
            timestep (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        """
        controller_params = {"delay": tau/dt, "max_deaccel": decel_max,
                             "noise": noise}
        BaseController.__init__(self, veh_id, controller_params)
        self.accel_queue = collections.deque()
        self.decel_max = decel_max
        self.accel_max = accel_max
        self.veh_id = veh_id
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta
        self.h_st = h_st
        self.h_go = h_go
        self.tau = tau
        self.dt = dt
        
    def get_accel(self, env):
        lead_id = env.vehicles.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.accel_max

        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)
        h = env.vehicles.get_headway(self.veh_id)
        h_dot = lead_vel - this_vel

        # V function here - input: h, output : Vh
        if h <= self.h_st:
            Vh = 0
        elif self.h_st < h < self.h_go:
            Vh = self.v_max / 2 * (1 - math.cos(math.pi * (h - self.h_st) /
                                                (self.h_go - self.h_st)))
        else:
            Vh = self.v_max

        acc = self.alpha*(Vh - this_vel) + self.beta*(h_dot)

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb
            # filling (currently), etc
            self.accel_queue.appendleft(acc)

        return max(min(self.accel_queue.pop(), self.accel_max),
                   -1 * abs(self.decel_max))

    def reset_delay(self, env):
        self.accel_queue.clear()


class LinearOVM(BaseController):

    def __init__(self, veh_id, v_max=30, accel_max=15, decel_max=-5,
                 adaptation=0.65, h_st=5, tau=0.5, dt=0.1, noise=0):
        """
        Instantiates a Linear OVM controller

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        v_max: float, optional
            max velocity (default: 30)
        accel_max: float, optional
            max acceleration (default: 15)
        decel_max: float, optional
            max deceleration (default: -5)
        adaptation: float
            adaptation constant (default: 0.65)
        h_st: float, optional
            headway for stopping (default: 5)
        tau: float, optional
            time delay (default: 0.5)
        dt: float, optional
            timestep (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        """
        controller_params = {"delay": tau / dt, "max_deaccel": decel_max,
                             "noise": noise}
        BaseController.__init__(self, veh_id, controller_params)
        self.accel_queue = collections.deque()
        self.decel_max = decel_max
        self.acc_max = accel_max
        self.veh_id = veh_id
        # 4.8*1.85 for case I, 3.8*1.85 for case II, per Nakayama
        self.v_max = v_max
        # TAU in Traffic Flow Dynamics textbook
        self.adaptation = adaptation
        self.h_st = h_st
        self.delay_time = tau
        self.dt = dt

    def get_accel(self, env):
        this_vel = env.vehicles.get_speed(self.veh_id)
        h = env.vehicles.get_headway(self.veh_id)

        # V function here - input: h, output : Vh
        alpha = 1.689  # the average value from Nakayama paper
        if h < self.h_st:
            Vh = 0
        elif self.h_st <= h <= self.h_st + self.v_max/alpha:
            Vh = alpha * (h - self.h_st)
        else:
            Vh = self.v_max

        acc = (Vh - this_vel) / self.adaptation

        while len(self.accel_queue) <= self.delay:
            # Some behavior here for initial states - extrapolation, dumb
            # filling (currently), etc
            self.accel_queue.appendleft(acc)

        return max(min(self.accel_queue.pop(), self.acc_max),
                   -1 * abs(self.decel_max))

    def reset_delay(self, env):
        self.accel_queue.clear()


class IDMController(BaseController):

    def __init__(self, veh_id, v0=30, T=1, a=1, b=1.5, delta=4, s0=2, s1=0,
                 decel_max=-5, dt=0.1, noise=0):
        """
        Instantiates an Intelligent Driver Model (IDM) controller

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        v0: float, optional
            desirable velocity, in m/s (default: 30)
        T: float, optional
            safe time headway, in s (default: 1)
        a: float, optional
            maximum acceleration, in m/s2 (default: 1)
        b: float, optional
            comfortable deceleration, in m/s2 (default: 1.5)
        delta: float, optional
            acceleration exponent (default: 4)
        s0: float, optional
            linear jam distance, in m (default: 2)
        s1: float, optional
            nonlinear jam distance, in m (default: 0)
        decel_max: float, optional
            max deceleration, in m/s2 (default: -5)
        dt: float, optional
            timestep, in s (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        """
        tau = T  # the time delay is taken to be the safe time headway
        controller_params = {"delay": tau / dt, "max_deaccel": decel_max,
                             "noise": noise}
        BaseController.__init__(self, veh_id, controller_params)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.s1 = s1
        self.max_deaccel = decel_max
        self.dt = dt

    def get_accel(self, env):
        this_vel = env.vehicles.get_speed(self.veh_id)
        lead_id = env.vehicles.get_leader(self.veh_id)
        h = env.vehicles.get_headway(self.veh_id)

        # negative headways may be registered by sumo at intersections/junctions
        # setting them to 0 causes vehicles to not move; therefore, we maintain
        # these negative headways to let sumo control the dynamics as it sees
        # fit at these points
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.vehicles.get_speed(lead_id)
            s_star = \
                self.s0 + max([0, this_vel*self.T + this_vel*(this_vel-lead_vel)
                               / (2 * np.sqrt(self.a * self.b))])

        return self.a * (1 - (this_vel/self.v0)**self.delta - (s_star/h)**2)

    def reset_delay(self, env):
        pass
