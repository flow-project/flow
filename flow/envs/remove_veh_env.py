"""This env removes vehicles that have just entered with some pre-defined frequency"""
import numpy as np
from flow.envs import TestEnv

class RemoveVehEnv(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.remove_freq = 2
        self.velocity_history = []
        self.look_back_len = 400

    def additional_command(self):
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('119257908#0') if self.k.vehicle.get_position(veh_id) < 150]
        entrance_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        self.velocity_history.append(entrance_veh_speed)
        if len(self.velocity_history) > self.look_back_len:
            if self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0 and self.time_counter % int(self.remove_freq) == 0:
                for veh_id in self.k.vehicle.get_departed_ids():
                    print('removing veh id ', veh_id)
                    self.k.vehicle.remove(veh_id)
            elif (self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0):
                for veh_id in self.k.vehicle.get_departed_ids():
                    self.k.vehicle.kernel_api.vehicle.slowDown(veh_id, self.velocity_history[-self.look_back_len], 1e-3)

