import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from flow.core.kernel.base_energy import BaseEnergyModel
from abc import ABCMeta, abstractmethod

class PowerDemandModel(BaseEnergyModel, metaclass=ABCMeta):
    """Calculate power consumption of a vehicle.
    Assumes vehicle is an average sized vehicle.
    The power calculated here is the lower bound of the 
    actual power consumed by the vehicle
    """

    def __init__(self,kernel):
        self.k = kernel
        self.g = 9.81
        self.rho_air = 1.225
        self.mass = 1200
        self.rolling_res_coeff = 0.005
        self.aerodynamic_drag_coeff = 0.3
        self.cross_area = 2.6
        self.gamma = 1
    
    def calculate_power(self, accel, speed, grade):
        accel_slope_forces = self.mass * speed * ((np.heaviside(accel, 0.5) * (1 - self.gamma) + self.gamma)) * accel
        accel_slope_forces += + self.g * math.sin(grade)
        rolling_friction = self.mass * self.g * self.rolling_res_coeff * speed
        air_drag = 0.5 * self.rho_air * self.cross_area * self.aerodynamic_drag_coeff * speed**3
        power = accel_slope_forces + rolling_friction + air_drag
        return power
      
    @abstractmethod
    def get_regen_cap(self, accel, speed, grade)
        pass

    def get_instantaneous_power(self, accel, speed, grade):
        power = max(get_regen_cap(), self.calculate_power(accel, speed, grade))
        return power


class PDMCombustionEngine(PowerDemandModel):
    
    # Power Demand Model for a combustion engine vehicle
    def get_regen_cap(self, accel, speed, grade)
        return 0

class PDMElectric(PowerDemandModel):
    
    # Power Demand Model for an electric vehicle
    def get_regen_cap(self, accel, speed, grade)
        return -2.8 * speed