from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
import numpy as np

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 3,
    "max_decel": 3,
    "lane_change_duration": 5,
    "target_velocity": 10,
}


class LaneChangeAccelEnv(Env):
    """Environment used to train autonomous vehicles to improve traffic flows
    when lane-change and acceleration actions are permitted by the rl agent.

    States
    ------
    The state consists of the velocities, absolute position, and lane index of
    all vehicles in the network. This assumes a constant number of vehicles.

    Actions
    -------
    Actions consist of:
    - a (continuous) acceleration from -abs(max_decel) to max_accel, specified
      in env_params
    - a (continuous) lane-change action from -1 to 1, used to determine the
      lateral direction the vehicle will take.
    Lane change actions are performed only if the vehicle has not changed lanes
    for the lane change duration specified in env_params.

    Rewards
    -------
    The reward function is the two-norm of the distance of the speed of the
    vehicles in the network from a desired speed, combined with a penalty to
    discourage excess lane changes by the rl vehicle.

    Termination
    -----------
    A rollout is terminated if the time horizon is reached or if two vehicles
    collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.
                               format(p))

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.vehicles.num_rl_vehicles
        ub = [max_accel, 1] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(low=-np.inf, high=np.inf,
                    shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        lane = Box(low=0, high=self.scenario.lanes-1,
                   shape=(self.vehicles.num_vehicles,),
                   dtype=np.float32)
        absolute_pos = Box(low=0., high=np.inf,
                           shape=(self.vehicles.num_vehicles,),
                           dtype=np.float32)
        return Tuple((speed, absolute_pos, lane))

    def compute_reward(self, state, rl_actions, **kwargs):
        # compute the system-level performance of vehicles from a velocity
        # perspective
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes
        for veh_id in self.vehicles.get_rl_ids():
            if self.vehicles.get_state(veh_id, "last_lc") == self.time_counter:
                reward -= 1

        return reward

    def get_state(self):
        return np.array([[self.vehicles.get_speed(veh_id),
                          self.vehicles.get_absolute_position(veh_id),
                          self.vehicles.get_lane(veh_id)]
                         for veh_id in self.sorted_ids])

    def _apply_rl_actions(self, actions):
        acceleration = actions[::2]
        direction = np.round(actions[1::2])

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)


class LaneChangeAccelPOEnv(Env):
    """Partially observable variant of LaneChangeAccelEnv.

    States
    ------
    The state consists of the velocities, absolute position, and lane index of
    all vehicles in the network. This assumes a constant number of vehicles.

    Actions
    -------
    Actions consist of:
    - a (continuous) acceleration from -abs(max_decel) to max_accel, specified
      in env_params
    - a (continuous) lane-change action from -1 to 1, used to determine the
      lateral direction the vehicle will take.
    Lane change actions are performed only if the vehicle has not changed lanes
    for the lane change duration specified in env_params.

    Rewards
    -------
    The reward function is the two-norm of the distance of the speed of the
    vehicles in the network from a desired speed, combined with a penalty to
    discourage excess lane changes by the rl vehicle.

    Termination
    -----------
    A rollout is terminated if the time horizon is reached or if two vehicles
    collide into one another.
    """

    @property
    def observation_space(self):
        num_lanes = max(self.scenario.num_lanes(edge)
                        for edge in self.scenario.get_edge_list())

        return Box(low=-float("inf"),
                   high=float("inf"),
                   shape=(4 * self.vehicles.num_rl_vehicles * num_lanes,),
                   dtype=np.float32)

    def get_state(self):
        return np.array([[self.vehicles.get_speed(veh_id),
                          self.vehicles.get_absolute_position(veh_id),
                          self.vehicles.get_lane(veh_id)]
                         for veh_id in self.sorted_ids])

    def additional_command(self):
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)
