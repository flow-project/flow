from abc import ABCMeta, abstractmethod
from flow.core.params import VehicleParams


class BaseEnergyModel(metaclass=ABCMeta):
    """Base energy model class.

    Calculate the instantaneous power consumption of a vehicle in
    the network.  It returns the power in Watts regardless of the 
    vehicle type: whether EV or Combustion Engine, Toyota Prius or Tacoma
    or non-Toyota vehicles. Non-Toyota vehicles are set by default 
    to be an averaged-size vehicle.
    """

    def __init__(self):
        vehicle = VehicleParams()

    @abstractmethod
    def get_instantaneous_power(self):
        """Calculate the instantaneous power consumption of a vehicle.
        
        Must be implemented by child classes.
        """
        pass
