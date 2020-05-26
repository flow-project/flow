import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from flow.energy.base_energy import BaseEnergyModel

class PowerDemandModel(BaseEnergyModel):
    """Calculate power consumption of a vehicle.
    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the 
    actual power consumed by the vehicle
    """

    def __init__(self,kernel):
        self.k = kernel
        self.g = 9.8
        self.rho_air = 1.225
        self.Mass = 1200
        self.rolling_res_coeff = 0.005
        self.aerodymnamic_drag_coeff = 0.3
        self.cross_A = 2.6
    
    def get_energy(self, veh_id, grade):
        pass
    

class PDM_CEV(PowerDemandModel):
    
    # Power Demand Model for a combustion engine vehicle

    def get_energy(self, veh_id, grade):
        speed = self.k.get_speed(veh_id)
        if veh_id in self.k.previous_speeds:
            old_speed = self.k.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.sim_step
        power = self.Mass*speed*(accel+self.g*math.sin(grade))+self.Mass*self.g*self.rolling_res_coeff*speed+0.5*self.rho_air*self.cross_A*self.aerodymnamic_drag_coeff*speed**3

        return power

class PDM_EV(PowerDemandModel):
    
    # Power Demand Model for an electric vehicle

    def get_energy(self, veh_id, grade):
        speed = self.k.get_speed(veh_id)
        if veh_id in self.k.previous_speeds:
            old_speed = self.k.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.sim_step
        power = self.Mass*speed*(accel+self.g*math.sin(grade))+self.Mass*self.g*self.rolling_res_coeff*speed+0.5*self.rho_air*self.cross_A*self.aerodymnamic_drag_coeff*speed**3
        power = max(0,power)

        return power
