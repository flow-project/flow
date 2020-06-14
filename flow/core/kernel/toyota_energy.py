import dill as pickle
import boto3
import botocore
import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from flow.core.kernel.base_energy import BaseEnergyModel
import os
from abc import ABCMeta, abstractmethod

class ToyotaModel(BaseEnergyModel):

    def __init__(self, kernel, filename=None):
        self.k = kernel

        # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted', filename,'file.pkl') #move to init
        with open('file.pkl','rb') as file:
            self.toyota_energy = pickle.load(file) #self.prius_energy
        # delete pickle file
        os.remove(file.pkl)
    
    @abstractmethod
    def get_instantaneous_power(self):
        pass


class PriusEnergy(ToyotaModel):
    
    def __init__(self, kernel):
        super(PriusEnergy, self).__init__(kernel, filename = 'prius_test.pkl')

    def get_instantaneous_power(self, parameter, accel, speed, grade):

        socdot = self.toyota_energy(parameter, accel, speed, grade)

        return socdot
            
class TacomaEnergy(ToyotaModel):

    def __init__(self, kernel):
        super(TacomaEnergy, self).__init__(kernel, filename = 'tacoma_test.pkl')

    def get_instantaneous_power(self, parameter, accel, speed, grade):
        
        fc = self.toyota_energy(accel, speed, grade) # returns instantaneous fuel consumption
        return fc
