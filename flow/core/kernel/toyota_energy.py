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

class ToyotaModel(BaseEnergyModel):

    def __init__(self,kernel):
        self.k = kernel

        # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted',filename,'file.pkl') #move to init
        with open('file.pkl','rb')as file:
            self.toyota_energy = pickle.load(file) #self.prius_energy
        # delete pickle file
        os.remove(file.pkl)

    def get_energy(self,veh_id,soc,grade):
        pass


class prius_energy(ToyotaModel):

    super.__init__(kernel,filename = 'prius_test.pkl')

    def get_energy(self,veh_id,soc,grade):

        speed = self.k.get_speed(veh_id)

        if veh_id in self.k.previous_speeds:
            old_speed = self.k.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.sim_step

        old_soc = self.k.get_soc(veh_id)
        grade = 0 

        socdot = self.toyota_energy(old_soc,speed,accel,grade)

        return socdot
            
class tacoma_energy(ToyotaModel):

    super.__init__(kernel,filename = 'tacoma_test.pkl')

    def get_energy(self,veh_id,grade, sim_step):
        
        speed = self.k.get_speed(veh_id)

        if veh_id in self.k.previous_speeds:
            old_speed = self.k.previous_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/self.sim_step
        grade = 0 

        fc = tacoma_test(speed,accel,grade) # returns instantanoes fuel consumption
        return fc


