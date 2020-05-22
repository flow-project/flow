import dill as pickle
import boto3
import botocore
import math
import random
import statistics
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple


class EnergyModel():
    """Base energy model class."""

    def __init__(self,kernel):
        self.vehicle = vehicle

    def get_energy(self, veh_id):
        #return self.calc_energy(veh_id)
        pass


class PowerDemandModel(EnergyModel):

    def __init__(self,kernel):
        self.k = kernel

        """Calculate power consumption of a vehicle.
        Assumes vehicle is an average sized vehicle.
        The power calculated here is the lower bound of the actual power consumed
        by a vehicle.
        M  mass of average sized vehicle (kg)
        g  gravitational acceleration (m/s^2)
        Cr rolling resistance coefficient
        Ca  aerodynamic drag coefficient
        rho  air density (kg/m^3)
        A  vehicle cross sectional area (m^2)
        """
        g = 9.8
        rho = 1.225
        
        if vtype == 'average':
             M = 1200
             Cr = 0.005
             Ca = 0.3
             A = 2.6

    def get_energy(self, veh_id)
        # if we know the constants for other vehicle types, we may put them here as well
        speed = self.k.get_speed(veh_id)
        if veh_id in self.k.old_speeds:
            old_speed = self.k.old_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/0.1
        power = M * speed * accel + M * g * Cr * speed + 0.5 * rho * A * Ca * speed ** 3

        return power


class ToyotaPriusEVModel(EnergyModel):

    def __init__(self,kernel):
        self.k = kernel

        # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted','prius_test.pkl','PriusEVModel.pkl') #move to init
        with open('PriusEVModel.pkl','rb')as file:
            prius_energy = pickle.load(file) #self.prius_energy
        # delete pickle file

    def get_energy(self):

        speed = self.k.get_speed(veh_id)

        if veh_id in self.k.old_speeds:
            old_speed = self.k.old_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/0.1

        old_soc = self.k.get_soc(veh_id)
        grade = 0 

        socdot = prius_energy(old_soc,speed,accel,grade)
        new_soc = old_soc + socdot*0.1

        return new_soc # return current soc level


class ToyotaTacomaModel(EnergyModel):

    def __init__(self,kernel):
        self.k = kernel

         # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted','tacoma_test.pkl','TacomaModel.pkl')

        with open('TacomaModel.pkl','rb')as file:
            tacoma_energy = pickle.load(file)


    def get_energy(self,speed,accel,grade):
        
        speed = self.k.get_speed(veh_id)

        if veh_id in self.k.old_speeds:
            old_speed = self.k.old_speeds[veh_id]
        else:
            old_speed = speed

        accel = (speed - old_speed)/0.1
        grade = 0 

        return tacoma_energy(speed,accel,grade) # returns instantanoes fuel consumption
