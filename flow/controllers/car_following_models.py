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
    """CFM controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : SumoCarFollowingParams
        see parent class
    k_d : float
        headway gain (default: 1)
    k_v : float
        gain on difference between lead velocity and current (default: 1)
    k_c : float
        gain on difference from desired velocity to current (default: 1)
    d_des : float
        desired headway (default: 1)
    v_des : float
        desired velocity (default: 8)
    time_delay : float, optional
        time delay (default: 0.0)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate a CFM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )

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

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    k_d : float
        gain on distances to lead/following cars (default: 1)
    k_v : float
        gain on vehicle velocity differences (default: 1)
    k_c : float
        gain on difference from desired velocity to current (default: 1)
    d_des : float
        desired headway (default: 1)
    v_des : float
        desired velocity (default: 8)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate a Bilateral car-following model controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )

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

class LACController(BaseController):
    """Linear Adaptive Cruise Control.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    k_1 : float
        design parameter (default: 0.8)
    k_2 : float
        design parameter (default: 0.9)
    h : float
        desired time gap  (default: 1.0)
    tau : float
        lag time between control input u and real acceleration a (default:0.1)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=0.3,
                 k_2=0.4,
                 h=1,
                 tau=0.1,
                 a=0,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate a Linear Adaptive Cruise controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.h = h
        self.tau = tau
        self.a = a

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        headway = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        ex = headway - L - self.h * this_vel
        ev = lead_vel - this_vel
        u = self.k_1*ex + self.k_2*ev
        a_dot = -(self.a/self.tau) + (u/self.tau)
        self.a = a_dot*env.sim_step + self.a

        return self.a

class OVMController(BaseController):
    """Optimal Vehicle Model controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    alpha : float
        gain on desired velocity to current velocity difference
        (default: 0.6)
    beta : float
        gain on lead car velocity and self velocity difference
        (default: 0.9)
    h_st : float
        headway for stopping (default: 5)
    h_go : float
        headway for full speed (default: 35)
    v_max : float
        max velocity (default: 30)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 alpha=1,
                 beta=1,
                 h_st=2,
                 h_go=15,
                 v_max=30,
                 time_delay=0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate an Optimal Vehicle Model controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
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
    """Linear OVM controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    v_max : float
        max velocity (default: 30)
    adaptation : float
        adaptation constant (default: 0.65)
    h_st : float
        headway for stopping (default: 5)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_max=30,
                 adaptation=0.65,
                 h_st=5,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate a Linear OVM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
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

# class ACC_Switched_Controller(BaseController):
#     """Adaptive Cruise Control with switch to Speed Control.

#     Attributes
#     ----------
#     veh_id : str
#         Vehicle ID for SUMO identification
#     car_following_params : flow.core.params.SumoCarFollowingParams
#         see parent class
#     k_1 : float
#         design parameter (default: 0.1)
#     k_2 : float
#         design parameter (default: 0.2)
#     h : float
#         desired time gap  (default: 1.0)
#     k_3 : float
#         gain for Cruise controller (default: 30)
#     V_m : float
#         Maximum Speed
#     time_delay : float
#         time delay (default: 0.5)
#     noise : float
#         std dev of normal perturbation to the acceleration (default: 0)
#     fail_safe : str
#         type of flow-imposed failsafe the vehicle should posses, defaults
#         to no failsafe (None)
#     """

#     def __init__(self,
#                  veh_id,
#                  car_following_params,
#                  k_1=0.1,
#                  k_2=0.2,
#                  k_3=0.2,
#                  V_m=30,
#                  h=1.2,
#                  d_min=8.0,
#                  time_delay=0.0,
#                  noise=0,
#                  fail_safe=None):
#         """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
#         BaseController.__init__(
#             self,
#             veh_id,
#             car_following_params,
#             delay=time_delay,
#             fail_safe=fail_safe,
#             noise=noise)

#         self.veh_id = veh_id
#         self.k_1 = k_1
#         self.k_2 = k_2
#         self.d_min = d_min
#         self.k_3 = k_3
#         self.V_m = V_m
#         self.h = h

#     def get_accel(self, env):
#         """See parent class."""
#         lead_id = env.k.vehicle.get_leader(self.veh_id)
#         v_l = env.k.vehicle.get_speed(lead_id)
#         v = env.k.vehicle.get_speed(self.veh_id)
#         s = env.k.vehicle.get_headway(self.veh_id)
#         L = env.k.vehicle.get_length(self.veh_id)
#         s = s - L

#         u = self.accel_func(v, v_l, s)

#         self.a = u

#         return self.a
        
#         # u = self.k_1*ex + self.k_2*ev
#         # a_dot = -(self.a/self.tau) + (u/self.tau)
#         # self.a = a_dot*env.sim_step + self.a

#         return self.a

#     def accel_func(self,v,v_l,s):

#         ex = s - v*self.h - self.d_min
#         ev = v_l - v
#         u=0.0

#         if(ex > self.h*self.V_m):
#             u = self.k_3*(self.V_m - v)
#         else:
#             u = self.k_1*ex+self.k_2*ev

#         return u

# class ACC_Switched_Controller_Attacked(BaseController):

#     def __init__(self,
#                  veh_id,
#                  car_following_params,
#                  k_1=0.1,
#                  k_2=0.2,
#                  k_3=0.2,
#                  V_m=30,
#                  h=1.2,
#                  d_min=8.0,
#                  switch_param_time=0.0,
#                  SS_Threshold=20,
#                  Total_Attack_Duration = 3.0,
#                  attack_decel_rate = -.45,
#                  time_delay=0.0,
#                  noise=0,
#                  fail_safe=None):
#         """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
#         BaseController.__init__(
#             self,
#             veh_id,
#             car_following_params,
#             delay=time_delay,
#             fail_safe=fail_safe,
#             noise=noise)

#         print('Attack Vehicle Spawned.')

#         self.veh_id = veh_id
#         self.k_1_congest = k_1
#         self.k_2_congest = k_2
#         self.k_1 = 1.0
#         self.k_2 = 1.0
#         self.d_min = d_min
#         self.k_3 = k_3
#         self.V_m = V_m
#         self.h = h
#         self.isUnderAttack = False
#         self.numSteps_Steady_State = 0
#         self.SS_Threshold = SS_Threshold #number seconds at SS to initiate attack
#         self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
#         self.Curr_Attack_Duration = 0.0 
#         self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
#         self.a = 0.0
#         self.switch_param_time = switch_param_time


#     def Attack_accel(self,env):
#         #Declerates the car for a set period at a set rate:

#         self.a = self.attack_decel_rate

#         self.Curr_Attack_Duration += env.sim_step

#         s = env.k.vehicle.get_headway(self.veh_id)
#         L = env.k.vehicle.get_length(self.veh_id)
#         s = s - L
#         v = env.k.vehicle.get_speed(self.veh_id) 

#         if(s < (v*self.h)):
#             #If vehicle in front is getting too close, break from disturbance
#             self.Reset_After_Attack(env)

#         if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
#             self.Reset_After_Attack(env)

#     def Reset_After_Attack(self,env):
#         self.isUnderAttack = False
#         self.numSteps_Steady_State = 0
#         self.Curr_Attack_Duration = 0.0
#         pos  = env.k.vehicle.get_position(self.veh_id)
#         lane = env.k.vehicle.get_lane(self.veh_id)
#         print('Attack Finished: '+str(self.veh_id))
#         print('Position of Attack: '+str(pos))
#         print('Lane of Attack: '+str(lane))

#     def Check_For_Steady_State(self):

#         if((self.a < .1) | (self.a > -.1)):
#             self.numSteps_Steady_State += 1
#         else:
#             self.numSteps_Steady_State = 0

#     def normal_ACC_accel(self,env):
#         lead_id = env.k.vehicle.get_leader(self.veh_id)
#         v_l = env.k.vehicle.get_speed(lead_id)
#         v = env.k.vehicle.get_speed(self.veh_id)
#         s = env.k.vehicle.get_headway(self.veh_id)
#         L = env.k.vehicle.get_length(self.veh_id)
#         s = s - L


#         u = self.accel_func(v, v_l, s)

#         self.a = u

#     def accel_func(self,v,v_l,s):

#         ex = s - v*self.h - self.d_min
#         ev = v_l - v
#         u=0.0

#         if(ex > self.h*self.V_m):
#             u = self.k_3*(self.V_m - v)
#         else:
#             u = self.k_1*ex+self.k_2*ev

#         return u

#     def Check_Start_Attack(self,env):
#         step_size = env.sim_step
#         SS_length = step_size * self.numSteps_Steady_State
#         if(SS_length >= self.SS_Threshold):
#             self.isUnderAttack = True
#         else:
#             self.isUnderAttack = False

#     def get_accel(self, env):
#         """See parent class."""
#         if(env.time_counter >= self.switch_param_time):
#             self.k_1 = self.k_1_congest
#             self.k_2 = self.k_2_congest

#         if(not self.isUnderAttack):
#             # No attack currently happening:
#             self.normal_ACC_accel(env)
#             # Check to see if driving near steady-state:
#             self.Check_For_Steady_State()
#             # Check to see if need to initiate attack:
#             self.Check_Start_Attack(env)
#             # Specificy that no attack is being executed:
#             env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)
#         else:
#             #Attack under way:
#             self.Attack_accel(env)
#             # Specify that an attack is happening:
#             env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)


#         return self.a

# class IDMController_Set_Congestion(BaseController):

#     def __init__(self,
#                  veh_id,
#                  v0=30,
#                  T=1,
#                  a=3.0,
#                  b=1.0,
#                  delta=4,
#                  s0=2,
#                  switch_param_time=0.0,
#                  time_delay=0.0,
#                  noise=0,
#                  fail_safe=None,
#                  display_warnings=True,
#                  car_following_params=None):
#         """Instantiate an IDM controller."""
#         BaseController.__init__(
#             self,
#             veh_id,
#             car_following_params,
#             delay=time_delay,
#             fail_safe=fail_safe,
#             noise=noise,
#             display_warnings=display_warnings,
#         )
#         self.v0 = v0
#         self.T = T
#         self.a_congest = a
#         self.b_congest = b
#         self.a = 3.0
#         self.b = 1.0
#         self.delta = delta
#         self.s0 = s0
#         self.switch_param_time = switch_param_time

#     def get_accel(self, env):
#         """See parent class."""
#         v = env.k.vehicle.get_speed(self.veh_id)
#         lead_id = env.k.vehicle.get_leader(self.veh_id)
#         s = env.k.vehicle.get_headway(self.veh_id)

#         if(env.time_counter >= self.switch_param_time):
#             #By default the parameters are set
#             self.a = self.a_congest
#             self.b = self.b_congest

#         return self.accel_func(s,v,lead_id)

#     def accel_func(s,v,lead_id):
#         if abs(h) < 1e-3:
#             h = 1e-3

#         if lead_id is None or lead_id == '':  # no car ahead
#             s_star = 0
#         else:
#             lead_vel = env.k.vehicle.get_speed(lead_id)
#             s_star = self.s0 + max(
#                 0, v * self.T + v * (v - lead_vel) /
#                 (2 * np.sqrt(self.a * self.b)))

#         return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)

class IDMController(BaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
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
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
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

class SimCarFollowingController(BaseController):
    """Controller whose actions are purely defined by the simulator.

    Note that methods for implementing noise and failsafes through
    BaseController, are not available here. However, similar methods are
    available through sumo when initializing the parameters of the vehicle.

    Usage: See BaseController for usage example.
    """

    def get_accel(self, env):
        """See parent class."""
        return None

class GippsController(BaseController):
    """Gipps' Model controller.

    For more information on this controller, see:
    Traffic Flow Dynamics written by M.Treiber and A.Kesting
    By courtesy of Springer publisher, http://www.springer.com

    http://www.traffic-flow-dynamics.org/res/SampleChapter11.pdf

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    acc : float
        max acceleration, in m/s2 (default: 1.5)
    b : float
        comfortable deceleration, in m/s2 (default: -1)
    b_l : float
        comfortable deceleration for leading vehicle , in m/s2 (default: -1)
    s0 : float
        linear jam distance for saftey, in m (default: 2)
    tau : float
        reaction time in s (default: 1)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params=None,
                 v0=30,
                 acc=1.5,
                 b=-1,
                 b_l=-1,
                 s0=2,
                 tau=1,
                 delay=0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate a Gipps' controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )

        self.v_desired = v0
        self.acc = acc
        self.b = b
        self.b_l = b_l
        self.s0 = s0
        self.tau = tau

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        v_l = env.k.vehicle.get_speed(
            env.k.vehicle.get_leader(self.veh_id))

        # get velocity dynamics
        v_acc = v + (2.5 * self.acc * self.tau * (
                1 - (v / self.v_desired)) * np.sqrt(0.025 + (v / self.v_desired)))
        v_safe = (self.tau * self.b) + np.sqrt(((self.tau**2) * (self.b**2)) - (
                self.b * ((2 * (h-self.s0)) - (self.tau * v) - ((v_l**2) / self.b_l))))

        v_next = min(v_acc, v_safe, self.v_desired)

        return (v_next-v)/env.sim_step

class BandoFTLController(BaseController):
    """Bando follow-the-leader controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    alpha : float
        gain on desired velocity to current velocity difference
        (default: 0.6)
    beta : float
        gain on lead car velocity and self velocity difference
        (default: 0.9)
    h_st : float
        headway for stopping (default: 5)
    h_go : float
        headway for full speed (default: 35)
    v_max : float
        max velocity (default: 30)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 alpha=.5,
                 beta=20,
                 h_st=2,
                 h_go=10,
                 v_max=32,
                 want_max_accel=False,
                 time_delay=0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True):
        """Instantiate an Bando controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.veh_id = veh_id
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta
        self.h_st = h_st
        self.h_go = h_go
        self.want_max_accel = want_max_accel

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            if self.want_max_accel:
                return self.max_accel

        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        return self.accel_func(v, v_l, s)

    def accel_func(self, v, v_l, s):
        """Compute the acceleration function."""
        v_h = self.v_max * ((np.tanh(s/self.h_st-2)+np.tanh(2))/(1+np.tanh(2)))
        s_dot = v_l - v
        u = self.alpha * (v_h - v) + self.beta * s_dot/(s**2)
        return u
