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

    Calculate the instantanoeus power consumption of a vehicle in
    the network.  It returns the power in Watts regardless of the 
    vehicle type: whether EV or Combustion Engine, Toyota Prius or Tacoma
    or non-Toyota vehicles. Non-Toyota vehicles are set by deafult 
    to be an averaged-size vehicle.
    
    """

    def __init__(self,kernel):
        self.vehicle = vehicle

    @abstractmethod
    def get_energy(self, veh_id,model_param,grade):
        #return self.calc_energy(veh_id)
        pass
