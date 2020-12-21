"""Script containing the base vehicle energy class."""
from abc import ABCMeta, abstractmethod


class BaseEnergyModel(metaclass=ABCMeta):
    """Base energy model class.

    Calculate the instantaneous power consumption of a vehicle in
    the network.  It returns the power in Watts regardless of the
    vehicle type: whether EV or Combustion Engine, Toyota Prius or Tacoma
    or non-Toyota vehicles. Non-Toyota vehicles are set by default
    to be an averaged-size vehicle.

    Note: road grade is included as an input parameter, but the
    functional dependence on road grade is not yet implemented.
    """

    def __init__(self):
        # 15 kilowatts = 1 gallon/hour conversion factor
        self.conversion = 15e3

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

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        """Calculate the instantaneous fuel consumption of a vehicle.

        Fuel consumption is reported in gallons per hour, with the conversion
        rate of 15kW = 1 gallon/hour.

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
        return self.get_instantaneous_power(accel, speed, grade) / self.conversion
