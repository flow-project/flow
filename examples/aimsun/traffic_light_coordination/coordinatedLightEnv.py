from flow.envs import TestEnv


class CoordinatedEnv(TestEnv):
    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        # veh_types = ["Car", "Car HOV", "Truck - Medium Duty (SU)"]
        # self.k.vehicle.tracked_vehicle_types.update(veh_types)
        # tl_ids = self.k.traffic_light.get_ids()
        # print(tl_ids)
        print(self.k.traffic_light.set_intersection_offset(3344, -20))
