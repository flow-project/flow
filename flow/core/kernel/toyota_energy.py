import dill as pickle
import boto3
import botocore
import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from flow.energy.base_energy import BaseEnergyModel
import os
from abc import ABCMeta, abstractmethod

class ToyotaModel(BaseEnergyModel):

    def __init__(self, kernel, filename=None):
        self.k = kernel
        self.k.env.vehicle = vehicle

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

    def get_instantaneous_power(self:

        speed = self.k.env.get_speed(veh_id)

        if veh_id in self.k.env.previous_speeds:
            old_speed = self.k.env.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.k.env.sim_step

        old_soc = self.k.env.get_soc(veh_id)
        grade = 0 

        socdot = self.toyota_energy(old_soc, speed, accel, grade)

        return socdot
            
class TacomaEnergy(ToyotaModel):

    def __init__(self, kernel):
        super(PriusEnergy, self).__init__(kernel, filename = 'tacoma_test.pkl')

    def get_instantaneous_power(self):
        
        speed = self.k.env.get_speed(veh_id)

        if veh_id in self.k.env.previous_speeds:
            old_speed = self.k.env.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.k.env.sim_step
        grade = 0 

        fc = self.toyota_energy(speed,accel,grade) # returns instantanoes fuel consumption
        return fc


