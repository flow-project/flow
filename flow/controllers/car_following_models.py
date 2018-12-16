"""
Contains several custom car-following control models.

These controllers can be used to modify the acceleration behavior of vehicles
in Flow to match various prominent car-following models that can be calibrated.

Each controller includes the function ``get_accel(self, env) -> acc`` which,
using the current state of the world and existing parameters, uses the control
model to return a vehicle acceleration.
"""
import math
import numpy as np

from flow.controllers.base_controller import BaseController


class CFMController(BaseController):
    """CFM controller."""

    def __init__(self,
                 veh_id,
                 sumo_cf_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a CFM controller.

        Attributes
        ----------
        veh_id : str
            Vehicle ID for SUMO identification
        sumo_cf_params : SumoCarFollowingParams
            see parent class
        k_d : float
            headway gain (default: 1)
        k_v : float, optional
            gain on difference between lead velocity and current (default: 1)
        k_c : float, optional
            gain on difference from desired velocity to current (default: 1)
        d_des : float, optional
            desired headway (default: 1)
        v_des : float, optional
            desired velocity (default: 8)
        time_delay : float, optional
            time delay (default: 0.0)
        noise : float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe : str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        d_l = env.k.vehicle.get_headway(self.veh_id)

        return self.k_d*(d_l - self.d_des) + self.k_v*(lead_vel - this_vel) + \
            self.k_c*(self.v_des - this_vel)


class BCMController(BaseController):
    """Bilateral car-following model controller.

    This model looks ahead and behind when computing its acceleration.
    """

    def __init__(self,
                 veh_id,
                 sumo_cf_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Bilateral car-following model controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        sumo_cf_params: SumoCarFollowingParams
            see parent class
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
        time_delay: float, optional
            time delay (default: 0.5)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des

    def get_accel(self, env):
        """See parent class.

        From the paper:
        There would also be additional control rules that take
        into account minimum safe separation, relative speeds,
        speed limits, weather and lighting conditions, traffic density
        and traffic advisories
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        trail_id = env.k.vehicle.get_follower(self.veh_id)
        trail_vel = env.k.vehicle.get_speed(trail_id)

        headway = env.k.vehicle.get_headway(self.veh_id)
        footway = env.k.vehicle.get_headway(trail_id)

        return self.k_d * (headway - footway) + \
            self.k_v * ((lead_vel - this_vel) - (this_vel - trail_vel)) + \
            self.k_c * (self.v_des - this_vel)


class OVMController(BaseController):
    """Optimal Vehicle Model controller."""

    def __init__(self,
                 veh_id,
                 sumo_cf_params,
                 alpha=1,
                 beta=1,
                 h_st=2,
                 h_go=15,
                 v_max=30,
                 time_delay=0,
                 noise=0,
                 fail_safe=None):
        """Instantiate an Optimal Vehicle Model controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        sumo_cf_params: SumoCarFollowingParams
            see parent class
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
        time_delay: float, optional
            time delay (default: 0.5)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.veh_id = veh_id
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta
        self.h_st = h_st
        self.h_go = h_go

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        h_dot = lead_vel - this_vel

        # V function here - input: h, output : Vh
        if h <= self.h_st:
            v_h = 0
        elif self.h_st < h < self.h_go:
            v_h = self.v_max / 2 * (1 - math.cos(math.pi * (h - self.h_st) /
                                                 (self.h_go - self.h_st)))
        else:
            v_h = self.v_max

        return self.alpha * (v_h - this_vel) + self.beta * h_dot


class LinearOVM(BaseController):
    """Linear OVM controller."""

    def __init__(self,
                 veh_id,
                 sumo_cf_params,
                 v_max=30,
                 adaptation=0.65,
                 h_st=5,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Linear OVM controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        sumo_cf_params: SumoCarFollowingParams
            see parent class
        v_max: float, optional
            max velocity (default: 30)
        adaptation: float
            adaptation constant (default: 0.65)
        h_st: float, optional
            headway for stopping (default: 5)
        time_delay: float, optional
            time delay (default: 0.5)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.veh_id = veh_id
        # 4.8*1.85 for case I, 3.8*1.85 for case II, per Nakayama
        self.v_max = v_max
        # TAU in Traffic Flow Dynamics textbook
        self.adaptation = adaptation
        self.h_st = h_st

    def get_accel(self, env):
        """See parent class."""
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # V function here - input: h, output : Vh
        alpha = 1.689  # the average value from Nakayama paper
        if h < self.h_st:
            v_h = 0
        elif self.h_st <= h <= self.h_st + self.v_max / alpha:
            v_h = alpha * (h - self.h_st)
        else:
            v_h = self.v_max

        return (v_h - this_vel) / self.adaptation


class IDMController(BaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 dt=0.1,
                 noise=0,
                 fail_safe=None,
                 sumo_cf_params=None):
        """Instantiate an IDM controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        sumo_cf_params: SumoCarFollowingParams
            see parent class
        v0: float, optional
            desirable velocity, in m/s (default: 30)
        T: float, optional
            safe time headway, in s (default: 1)
        b: float, optional
            comfortable deceleration, in m/s2 (default: 1.5)
        delta: float, optional
            acceleration exponent (default: 4)
        s0: float, optional
            linear jam distance, in m (default: 2)
        dt: float, optional
            timestep, in s (default: 0.1)
        noise: float, optional
            std dev of normal perturbation to the acceleration (default: 0)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.dt = dt

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # negative headways may be registered by sumo at intersections/
        # junctions. Setting them to 0 causes vehicles to not move; therefore,
        # we maintain these negative headways to let sumo control the dynamics
        # as it sees fit at these points.
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)


class SumoCarFollowingController(BaseController):
    """Controller whose actions are purely defined by sumo.

    Note that methods for implementing noise and failsafes through
    BaseController, are not available here. However, similar methods are
    available through sumo when initializing the parameters of the vehicle.
    """

    def __init__(self, veh_id, sumo_cf_params):
        """Instantiate a sumo controller.

        Attributes
        ----------
        veh_id: str
            name of the vehicle
        sumo_cf_params: SumoCarFollowingParams
            see parent class
        """
        super().__init__(veh_id, sumo_cf_params)
        self.sumo_controller = True

    def get_accel(self, env):
        """See parent class."""
        return None
