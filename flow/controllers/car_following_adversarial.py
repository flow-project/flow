import math
import numpy as np

from flow.controllers.base_controller import BaseController

class ACC_Switched_Controller(BaseController):
    """Adaptive Cruise Control with switch to Speed Control.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    k_1 : float
        design parameter (default: 0.1)
    k_2 : float
        design parameter (default: 0.2)
    h : float
        desired time gap  (default: 1.0)
    k_3 : float
        gain for Cruise controller (default: 30)
    V_m : float
        Maximum Speed
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
                 k_1=0.1,
                 k_2=0.2,
                 k_3=0.2,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.d_min = d_min
        self.k_3 = k_3
        self.V_m = V_m
        self.h = h

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L

        u = self.accel_func(v, v_l, s)

        self.a = u

        return self.a
        
        # u = self.k_1*ex + self.k_2*ev
        # a_dot = -(self.a/self.tau) + (u/self.tau)
        # self.a = a_dot*env.sim_step + self.a

        return self.a

    def accel_func(self,v,v_l,s):

        ex = s - v*self.h - self.d_min
        ev = v_l - v
        u=0.0

        if(ex > self.h*self.V_m):
            u = self.k_3*(self.V_m - v)
        else:
            u = self.k_1*ex+self.k_2*ev

        return u

class ACC_Switched_Controller_Attacked(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=0.1,
                 k_2=0.2,
                 k_3=0.2,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 switch_param_time=0.0,
                 SS_Threshold=30,
                 Total_Attack_Duration = 3.0,
                 attack_decel_rate = -.8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        print('Attack Vehicle Spawned.')

        self.veh_id = veh_id
        self.k_1_congest = k_1
        self.k_2_congest = k_2
        self.k_1 = 1.0
        self.k_2 = 1.0
        self.d_min = d_min
        self.k_3 = k_3
        self.V_m = V_m
        self.h = h
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.SS_Threshold = SS_Threshold #number seconds at SS to initiate attack
        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0 
        self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
        self.a = 0.0
        self.switch_param_time = switch_param_time
        self.switched_from_stable = False


    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:

        self.a = self.attack_decel_rate

        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id) 

        if(s < (v*(self.h-.2))):
            #If vehicle in front is getting too close, break from disturbance
            self.Reset_After_Attack(env)

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.Curr_Attack_Duration = 0.0
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        print('Attack Finished: '+str(self.veh_id))
        print('Position of Attack: '+str(pos))
        print('Lane of Attack: '+str(lane))

    def Check_For_Steady_State(self):

        if((self.a < .1) | (self.a > -.1)):
            self.numSteps_Steady_State += 1
        else:
            self.numSteps_Steady_State = 0

    def normal_ACC_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L


        u = self.accel_func(v, v_l, s)

        self.a = u

    def accel_func(self,v,v_l,s):

        ex = s - v*self.h - self.d_min
        ev = v_l - v
        u=0.0

        if(ex > self.h*self.V_m):
            u = self.k_3*(self.V_m - v)
        else:
            u = self.k_1*ex+self.k_2*ev

        return u

    def Check_Start_Attack(self,env):
        step_size = env.sim_step
        SS_length = step_size * self.numSteps_Steady_State
        if(SS_length >= self.SS_Threshold):
            self.isUnderAttack = True
        else:
            self.isUnderAttack = False

    def get_accel(self, env):
        """See parent class."""
        if((env.time_counter >= self.switch_param_time) & (not self.switched_from_stable)):
            #Check to see if should be in string-unstable regime
            self.k_1 = self.k_1_congest
            self.k_2 = self.k_2_congest
            self.switched_from_stable = True
            print('Switching to original model: '+self.veh_id)

        if(not self.isUnderAttack):
            # No attack currently happening:
            self.normal_ACC_accel(env)
            # Check to see if driving near steady-state:
            self.Check_For_Steady_State()
            # Check to see if need to initiate attack:
            self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)
        else:
            #Attack under way:
            self.Attack_accel(env)
            # Specify that an attack is happening:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)


        return self.a

class IDMController_Set_Congestion(BaseController):

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=3.0,
                 b=1.0,
                 delta=4,
                 s0=2,
                 switch_param_time=0.0,
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
        self.a_congest = a
        self.b_congest = b
        self.a = 3.0
        self.b = 1.0
        self.delta = delta
        self.s0 = s0
        self.switch_param_time = switch_param_time

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)

        if(env.time_counter >= self.switch_param_time):
            #By default the parameters are set
            self.a = self.a_congest
            self.b = self.b_congest

        return self.accel_func(s,v,lead_id,env)

    def accel_func(self,s,v,lead_id,env):
        if abs(s) < 1e-3:
            s = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / s)**2)


