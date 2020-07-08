"""Script containing the vehicle power demand model energy classes."""
import math
import numpy as np
from flow.energy_models.base_energy import BaseEnergyModel
from abc import ABCMeta, abstractmethod


class PowerDemandModel(BaseEnergyModel, metaclass=ABCMeta):
    """Vehicle Power Demand base energy model class.

    Calculate power consumption of a vehicle based on physics
    derivation. Assumes some vehicle characteristics. The
    power calculated here is the lower bound of the actual
    power consumed by the vehicle plus a bilinear polynomial
    function used as a correction factor.
    """

    def __init__(self,
                 kernel,
                 mass=2041,
                 area=3.2,
                 rolling_res_coeff=0.0027,
                 aerodynamic_drag_coeff=0.4,
                 p1_correction=4598.7155,
                 p3_correction=975.12719):
        self.k = kernel
        self.g = 9.807
        self.rho_air = 1.225
        self.gamma = 1
        self.mass = mass
        self.cross_area = area
        self.rolling_res_coeff = rolling_res_coeff
        self.aerodynamic_drag_coeff = aerodynamic_drag_coeff
        self.p1_correction = p1_correction
        self.p3_correction = p3_correction

    def calculate_power_at_the_wheels(self, accel, speed, grade):
        """Calculate the instantaneous power required.

        Parameters
        ----------
        accel : float
            Instantaneous acceleration of the vehicle
        speed : float
            Instantaneous speed of the vehicle
        grade : float
            Instantaneous road grade of the vehicle
        Returns
        -------
        float
        """
        accel_slope_forces = self.mass * speed * ((np.heaviside(accel, 0.5) * (1 - self.gamma) + self.gamma)) * accel
        accel_slope_forces += self.g * math.sin(grade)
        rolling_friction = self.mass * self.g * self.rolling_res_coeff * speed
        air_drag = 0.5 * self.rho_air * self.cross_area * self.aerodynamic_drag_coeff * speed**3
        power = accel_slope_forces + rolling_friction + air_drag
        return power

    @abstractmethod
    def get_regen_cap(self, accel, speed, grade):
        """Set the maximum power retainable from regenerative braking.

        A negative regen cap is interpretted as a positive regenerative power.

        Parameters
        ----------
        accel : float
            Instantaneous acceleration of the vehicle
        speed : float
            Instantaneous speed of the vehicle
        grade : float
            Instantaneous road grade of the vehicle
        Returns
        -------
        float
        """
        pass

    def get_power_correction_factor(self, accel, speed, grade):
        """Calculate the instantaneous power correction of a vehicle.

        Parameters
        ----------
        accel : float
            Instantaneous acceleration of the vehicle
        speed : float
            Instantaneous speed of the vehicle
        grade : float
            Instantaneous road grade of the vehicle
        Returns
        -------
        float
        """
        return self.p1_correction * accel + self.p3_correction * accel * speed

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class.

        Apply the regenerative braking cap to the modelled power demand.
        """
        regen_cap = self.get_regen_cap(accel, speed, grade)
        power_at_the_wheels = max(regen_cap, self.calculate_power_at_the_wheels(accel, speed, grade))
        correction_factor = max(regen_cap, self.get_power_correction_factor(accel, speed, grade))
        return power_at_the_wheels + correction_factor


class PDMCombustionEngine(PowerDemandModel):
    """Power Demand Model for a combustion engine vehicle."""

    def get_regen_cap(self, accel, speed, grade):
        """See parent class."""
        return 0


class PDMElectric(PowerDemandModel):
    """Power Demand Model for an electric vehicle."""

    def __init__(self, kernel):
        super(PDMElectric, self).__init__(kernel,
                                          mass=1663,
                                          area=2.4,
                                          rolling_res_coeff=0.007,
                                          aerodynamic_drag_coeff=0.24)

    def get_regen_cap(self, accel, speed, grade):
        """See parent class."""
        return -2.8 * speed
