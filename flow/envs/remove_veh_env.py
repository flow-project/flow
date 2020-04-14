"""This env removes vehicles that have just entered with some pre-defined frequency"""
import numpy as np
from flow.envs import TestEnv


class RemoveVehEnv(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.remove_freq = 1.5
        self.velocity_history = []
        self.look_back_len = 100

    def additional_command(self):
        # vehicles on the first edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('119257914') if self.k.vehicle.get_position(veh_id) < 150]
        entrance_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        # vehicles on the second edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('119257908#0') if self.k.vehicle.get_position(veh_id) < 150]
        second_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        # the speed at inflow should be delayed, since the congestion is travelling upstream
        self.velocity_history.append(min(second_lane_veh_speed, entrance_lane_veh_speed))
        # self.velocity_history.append(second_lane_veh_speed)
        if len(self.velocity_history) > self.look_back_len:
            if self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0 and int(self.time_counter % self.remove_freq) == 0:
                for veh_id in self.k.vehicle.get_departed_ids():
                    print('removing veh id ', veh_id)
                    self.k.vehicle.remove(veh_id)
            elif (self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0):
                for veh_id in self.k.vehicle.get_departed_ids():
                    self.k.vehicle.kernel_api.vehicle.slowDown(veh_id, self.velocity_history[-self.look_back_len], 1e-3)


class RemoveVehEnv2(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.remove_freq = 4
        self.velocity_history = []
        self.look_back_len = 25

    def additional_command(self):
        # vehicles on the first edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('highway_0') if self.k.vehicle.get_position(veh_id) < 150]
        entrance_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        # vehicles on the second edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('highway_1') if self.k.vehicle.get_position(veh_id) < 150]
        second_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        # the speed at inflow should be delayed, since the congestion is travelling upstream
        print('entrance lane speed ', entrance_lane_veh_speed)
        print('second lane speed ', second_lane_veh_speed)

        self.velocity_history.append(min(second_lane_veh_speed, entrance_lane_veh_speed))
        # self.velocity_history.append(second_lane_veh_speed)
        if len(self.velocity_history) > self.look_back_len:
            # don't remove until we hit congestion
            print(self.velocity_history[-1])
            if self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0 and int(self.time_counter % self.remove_freq) == 0:
                for veh_id in self.k.vehicle.get_departed_ids():
                    print('removing veh id ', veh_id)
                    self.k.vehicle.remove(veh_id)
            elif (self.velocity_history[-1] < 10.0 and self.velocity_history[-1] > 0.0):
                for veh_id in self.k.vehicle.get_departed_ids():
                    self.k.vehicle.kernel_api.vehicle.slowDown(veh_id, self.velocity_history[-self.look_back_len], 1e-3)


class RemoveVehEnv3(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.remove_freq = 4
        self.velocity_history = []
        self.look_back_len = 100
        self.lower_inflow_steps = 400
        self.counter = 0

    def additional_command(self):
        # vehicles on the first edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('119257914') if self.k.vehicle.get_position(veh_id) < 150]
        entrance_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))
        # vehicles on the second edge, checking for congestion there
        veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids_by_edge('119257908#0') if self.k.vehicle.get_position(veh_id) < 150]
        second_lane_veh_speed = np.nan_to_num(np.mean(self.k.vehicle.get_speed(veh_ids)))

        # store speed difference
        self.velocity_history.append(second_lane_veh_speed - entrance_lane_veh_speed)
        if len(self.velocity_history) == self.look_back_len + 1:
            self.velocity_history.pop(0)

        # when mean speed difference is positive (entrance speed is less than the second edge speed),
        # we start reducing demand for at least lower_inflow_steps
        if np.mean(self.velocity_history) > 0:
            self.counter = self.lower_inflow_steps

        if self.counter > 0 and int(self.time_counter % self.remove_freq) != 0:
            for veh_id in self.k.vehicle.get_departed_ids():
                print('removing veh id ', veh_id, self.time_counter)
                self.k.vehicle.remove(veh_id)
                self.counter -= 1
