"""Environment testing non_RL scenario 1 of the bayesian envs."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.envs.ring.accel import AccelEnv
from flow.core.kernel import Kernel

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}

class Bayesian1Env(AccelEnv):
    """Specify the starting positions and routes of 3 cars and 1 pedestrian"""

    def setup_initial_state(self):
        # generate starting position for the 3 vehicles in the network
        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        # save the initial state. This is used in the _reset function
        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            lane = start_lanes[i]
            speed = self.k.vehicle.get_initial_speed(veh_id)
            edge = start_pos[i][0]
            self.initial_state[veh_id] = (type_id, edge, lane, pos, speed)

