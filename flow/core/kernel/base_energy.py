import dill as pickle
import boto3
import botocore
import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from abc import ABCMeta, abstractmethod


class BaseEnergyModel(metaclass=ABCMeta):
    """Base energy model class.

    Calculate the instantaneous power consumption of a vehicle in
    the network.  It returns the power in Watts regardless of the 
    vehicle type: whether EV or Combustion Engine, Toyota Prius or Tacoma
    or non-Toyota vehicles. Non-Toyota vehicles are set by default 
    to be an averaged-size vehicle.
    """

    def __init__(self,kernel):
        self.vehicle = vehicle

    @abstractmethod
    def get_instantaneous_power(self, veh_id, model_param, grade):
        """Calculate the instantaneous power consumption of a vehicle.
        
        Must be implemented by child classes.
        """
        pass
