"""Script containing the Toyota energy classes."""
from abc import ABCMeta, abstractmethod
import dill as pickle
import boto3
import os

from flow.energy_models.base_energy import BaseEnergyModel


class ToyotaModel(BaseEnergyModel, metaclass=ABCMeta):
    """Base Toyota Energy model class."""

    def __init__(self, filename):
        super(ToyotaModel, self).__init__()

        # download file from s3 bucket
        s3 = boto3.client('s3')
        s3.download_file('toyota.restricted', filename, 'temp.pkl')

        with open('temp.pkl', 'rb') as file:
            try:
                self.toyota_energy = pickle.load(file)
                # delete pickle file
                os.remove('temp.pkl')
            except TypeError:
                print('Must use Python version 3.6.8 to unpickle')
                # delete pickle file
                os.remove('temp.pkl')
                raise

    @abstractmethod
    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        pass


class PriusEnergy(ToyotaModel):
    """Toyota Prius (EV) energy model class."""

    def __init__(self, sim_step, soc=0.9):
        super(PriusEnergy, self).__init__(filename='prius_ev.pkl')
        self.sim_step = sim_step
        self.soc = soc

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        socdot = self.toyota_energy(self.soc, accel, speed, grade)
        self.soc -= socdot * self.sim_step
        # FIXME (Joy): convert socdot to power
        return socdot


class TacomaEnergy(ToyotaModel):
    """Toyota Tacoma energy model class."""

    def __init__(self):
        super(TacomaEnergy, self).__init__(filename='tacoma.pkl')

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        return self.get_instantaneous_fuel_consumption(accel, speed, grade) * self.conversion

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        """See parent class."""
        fc = self.toyota_energy(accel, speed, grade)
        return fc * 3600.0 / 3217.25
