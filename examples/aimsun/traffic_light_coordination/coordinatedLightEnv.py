from flow.envs import TestEnv


class CoordinatedEnv(TestEnv):
    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        tl_ids = self.k.traffic_light.get_ids()
        print(tl_ids)
        print(self.k.traffic_light.set_intersection_offset(3344, 20))
