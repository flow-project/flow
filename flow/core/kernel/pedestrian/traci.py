
from flow.core.kernel.pedestrian import KernelPedestrian
import traci.constants as tc
import numpy as np

class TraCIPedestrian(KernelPedestrian):

    def __init__(self,
            master_kernel):

        KernelPedestrian.__init__(self, master_kernel)

        self.__ids = [] # ids of all pedestrians

    def get_speed(self, ped_id, error=-1001):
        if isinstance(ped_id, (list, np.ndarray)):
            return [self.get_speed(pedID, error) for pedID in ped_id]
        return self.kernel_api.person.getSpeed(ped_id)

    def get_position(self, ped_id, error=-1001):
        if isinstance(ped_id, (list, np.ndarray)):
            return [self.get_position(pedID, error) for pedID in ped_id]
        return self.kernel_api.person.getPosition(ped_id)

    def get_edge(self, ped_id, error=-1001):
        if isinstance(ped_id, (list, np.ndarray)):
            return [self.get_edge(pedID, error) for pedID in ped_id]
        return self.kernel_api.person.getRoadID(ped_id)
