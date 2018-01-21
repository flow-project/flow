
from flow.envs.loop_accel import SimpleAccelerationEnvironment

import numpy as np

class TrainedPolicyEnv(SimpleAccelerationEnvironment):

    gain_dif_vel = 0.5

    def compute_reward(self, state, rl_actions, **kwargs):
        desired_vel = np.array([self.env_params.additional_params["target_velocity"]] * self.vehicles.num_vehicles)
        curr_vel = np.array(self.vehicles.get_speed())
        diff_vel = np.linalg.norm(desired_vel - curr_vel)

        accel = self.vehicles.get_accel(veh_id="all")

        deaccel = [min(0, x) for x in accel]
        # print(deaccel)

        return np.linalg.norm(deaccel)