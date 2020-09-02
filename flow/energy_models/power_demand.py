"""Script containing the vehicle power demand model energy classes."""
from abc import ABCMeta
import numpy as np

from flow.energy_models.base_energy import BaseEnergyModel


class PowerDemandModel(BaseEnergyModel, metaclass=ABCMeta):
    """Vehicle Power Demand base energy model class.

    Calculate power consumption of a vehicle based on physics
    derivation. Assumes some vehicle characteristics. The
    power calculated here is the lower bound of the actual
    power consumed by the vehicle plus a bilinear polynomial
    function used as a correction factor.
    """

    def __init__(self,
                 mass=2041,
                 idle_coeff=3405.5481762,
                 linear_friction_coeff=83.123929917,
                 quadratic_friction_coeff=6.7650718327,
                 drag_coeff=0.7041355229,
                 p0_correction=0,
                 p1_correction=0,
                 p2_correction=0,
                 p3_correction=0):
        super(PowerDemandModel, self).__init__()

        self.mass = mass
        self.phys_power_coeffs = np.array([idle_coeff,
                                           linear_friction_coeff,
                                           quadratic_friction_coeff,
                                           drag_coeff])
        self.power_correction_coeffs = np.array([p0_correction,
                                                 p1_correction,
                                                 p2_correction,
                                                 p3_correction])

    def calculate_phys_power(self, accel, speed, grade):
        """Calculate the instantaneous power from physics-based derivation.

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
        state_variables = np.array([1, speed, speed**2, speed**3])
        power_0 = np.dot(self.phys_power_coeffs, state_variables)
        return self.mass * accel * speed + power_0

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
        state_variables = np.array([1, accel, speed, accel * speed])
        return max(0, np.dot(self.power_correction_coeffs, state_variables))

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        phys_power = self.calculate_phys_power(accel, speed, grade)
        power_correction_factor = self.get_power_correction_factor(accel, speed, grade)
        return phys_power + power_correction_factor


class PDMCombustionEngine(PowerDemandModel):
    """Power Demand Model for a combustion engine vehicle.

    For more information, see docs/Tacoma_EnergyModel.pdf
    """

    def __init__(self,
                 mass=2041,
                 idle_coeff=3405.5481762,
                 linear_friction_coeff=83.123929917,
                 quadratic_friction_coeff=6.7650718327,
                 drag_coeff=0.7041355229,
                 p1_correction=4598.7155,
                 p3_correction=975.12719):
        super(PDMCombustionEngine, self).__init__(mass=mass,
                                                  idle_coeff=idle_coeff,
                                                  linear_friction_coeff=linear_friction_coeff,
                                                  quadratic_friction_coeff=quadratic_friction_coeff,
                                                  drag_coeff=drag_coeff,
                                                  p1_correction=p1_correction,
                                                  p3_correction=p3_correction)

    def calculate_phys_power(self, accel, speed, grade):
        """See parent class."""
        return max(super(PDMCombustionEngine, self).calculate_phys_power(accel, speed, grade), 0)


class PDMElectric(PowerDemandModel):
    """Power Demand Model for an electric vehicle.

    For more information, see docs/Prius_EnergyModel.pdf
    """

    def __init__(self,
                 mass=1663,
                 idle_coeff=1.046,
                 linear_friction_coeff=119.166,
                 quadratic_friction_coeff=0.337,
                 drag_coeff=0.383,
                 p3_correction=296.66,
                 alpha=0.869,
                 beta=2338):
        super(PDMElectric, self).__init__(mass=mass,
                                          idle_coeff=idle_coeff,
                                          linear_friction_coeff=linear_friction_coeff,
                                          quadratic_friction_coeff=quadratic_friction_coeff,
                                          drag_coeff=drag_coeff,
                                          p3_correction=p3_correction)
        self.alpha = alpha
        self.beta = beta

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        mod_power = super(PDMElectric, self).get_instantaneous_power(accel, speed, grade)
        return max(mod_power, self.alpha * mod_power, -self.beta * speed)
