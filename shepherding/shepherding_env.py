
from flow.envs.loop_accel import SimpleAccelerationEnvironment

import numpy as np

class ShepherdingEnv(SimpleAccelerationEnvironment):

    gain_dif_vel = 0.5



    def __init__(self, env_params, sumo_params, scenario):
        SimpleAccelerationEnvironment.__init__(self, env_params, sumo_params, scenario)
        self.aggressive_headway_impatience = 1.0


    def compute_reward(self, state, rl_actions, **kwargs):
        desired_vel = np.array([self.env_params.additional_params["target_velocity"]] * self.vehicles.num_vehicles)
        curr_vel = np.array(self.vehicles.get_speed())
        diff_vel = np.linalg.norm(desired_vel - curr_vel)
        accel = self.vehicles.get_accel(veh_id="all")
        deaccel =  np.linalg.norm([min(0, x) for x in accel])
        print(diff_vel, deaccel)
        return -(0.5 * diff_vel + 0.5 * deaccel)