"""Script containing the base vehicle energy class."""
from abc import ABCMeta, abstractmethod


class BaseEnergyModel(metaclass=ABCMeta):
    """Base energy model class.

    Calculate the instantaneous power consumption of a vehicle in
    the network.  It returns the power in Watts regardless of the
    vehicle type: whether EV or Combustion Engine, Toyota Prius or Tacoma
    or non-Toyota vehicles. Non-Toyota vehicles are set by default
    to be an averaged-size vehicle.
    """

    def __init__(self, kernel):
        self.k = kernel

    @abstractmethod
    def get_instantaneous_power(self, accel, speed, grade):
        """Calculate the instantaneous power consumption of a vehicle.

        Must be implemented by child classes.

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
