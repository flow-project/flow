"""Script containing the Toyota energy classes."""
import dill as pickle
import boto3
from flow.energy_models.base_energy import BaseEnergyModel
import os
from abc import ABCMeta, abstractmethod


class ToyotaModel(BaseEnergyModel, metaclass=ABCMeta):
    """Base Toyota Energy model class."""

    def __init__(self, kernel, filename=None):
        self.k = kernel

        # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted', filename, 'temp.pkl')
        with open('temp.pkl', 'rb') as file:
            self.toyota_energy = pickle.load(file)

        # delete pickle file
        os.remove(file.pkl)

    @abstractmethod
    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        pass


class PriusEnergy(ToyotaModel):
    """Toyota Prius (EV) energy model class."""

    def __init__(self, kernel, soc=0.9):
        super(PriusEnergy, self).__init__(kernel, filename='prius_test.pkl')
        self.soc = soc

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        socdot = self.toyota_energy(self.soc, accel, speed, grade)
        self.soc -= socdot * self.k.env.sim_step
        return socdot


class TacomaEnergy(ToyotaModel):
    """Toyota Tacoma energy model class."""

    def __init__(self, kernel):
        super(TacomaEnergy, self).__init__(kernel, filename='tacoma_test.pkl')

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        fc = self.toyota_energy(accel, speed, grade)
        return fc
