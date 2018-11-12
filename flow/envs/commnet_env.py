import numpy as np
from gym.spaces.box import Box

from flow.core import rewards
from flow.envs.base_env import Env


class CommNetEnv(Env):
    """Environment used to train traffic lights to regulate traffic flow
    through an n x m grid.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL
      options are "controlled" and "actuated"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sumo_params, scenario):

        self.grid_array = scenario.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols

        super().__init__(env_params, sumo_params, scenario)

        # Saving env variables for plotting
        self.node_mapping = scenario.get_node_mapping()

        for i in range(self.rows * self.cols):
            self.traci_connection.trafficlight.setRedYellowGreenState(
                'center' + str(i), "GGGGGGGGGGGG")

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-1,
            high=1,
            shape=(100,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        # headway to vehicle in front, speed of vehicle in front
        # ego vehicle speed
        return Box(
            low=0,
            high=1,
            shape=(3 * 100,),
            dtype=np.float32)

    def get_state(self, rl_actions=None):
        """See class definition."""
        # compute the normalizers
        state_array = np.zeros((3, 100))
        for i, rl_id in enumerate(self.vehicles.get_rl_ids()):
            headway = self.vehicles.get_headway(rl_id)
            leader = self.vehicles.get_leader(rl_id)
            leader_speed = self.vehicles.get_speed(leader)
            ego_speed = self.vehicles.get_speed(rl_id)
            state_array[i] = np.concatenate(([headway / 1000.0],
                                             [leader_speed / 100.0],
                                             [ego_speed / 100.0]))
        return state_array

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        zero_mask = np.isclose(rl_actions, 0.0)
        rl_actions = rl_actions[not zero_mask]
        self.apply_acceleration(self.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, state, rl_actions, **kwargs):
        """See class definition."""
        return rewards.min_delay(self)
