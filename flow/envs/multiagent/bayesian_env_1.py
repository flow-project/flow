"""Environment testing scenario one of the bayesian envs."""
import numpy as np
from gym.spaces.box import Box
from flow.core.rewards import desired_velocity
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # how many objects in our local radius we want to return
    "max_object_num": 3,
    # how large of a radius to search in for a given vehicle in meters
    "search_radius": 20
}


class BayesianEnv1(MultiEnv):
    """Testing whether an agent can learn to navigate successfully crossing the env described
    in scenario 1 of Jakob's diagrams. Please refer to the sketch for more details. Basically,
    inferring that the human is going to cross allows one of the vehicles to succesfully cross.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        TBD

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters.

    Rewards
        TBD

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.observation_names = ["rel_x, rel_y, speed, is_ped, yaw"]
        self.search_radius = self.env_params.additional_params["search_radius"]

    @property
    def observation_space(self):
        """See class definition."""
        max_objects = self.env_params.additional_params["max_num_objects"]
        # the items per object are relative X, relative Y, speed, whether it is a pedestrian, and its yaw
        return Box(-float('inf'), float('inf'), shape=(max_objects * len(self.observation_names),), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                self.k.vehicle.apply_acceleration(rl_id, accel)

    def get_state(self):
        """For a radius around the car, return the 3 closest objects with their X, Y position relative to you,
        their speed, aflag indicating if they are a pedestrian or not, and their yaw."""

        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            # TODO add get x y as something that we store from TraCI
            # TODO(@nliu)
            observation = np.zeros(self.observation_space.shape[0])
            visible_ids = self.find_visible_objects(rl_id, self.search_radius)
            veh_x, veh_y = self.k.vehicle.get_x_y(rl_id)
            for index, obj_id in enumerate(visible_ids):
                x, y = self.k.vehicle.get_x_y(obj_id)
                rel_x = veh_x - x
                rel_y = veh_y - y
                # TODO(@nliu)
                # TODO add a check for whether an object is a pedestrian
                is_ped = self.k.pedestrian.is_pedestrian(obj_id)
                if is_ped:
                    # TODO(@nliu) is this even possible?
                    # TODO implement yaw checking
                    speed = self.k.pedestrian.get_speed(obj_id)
                    yaw = self.k.pedestrian.get_yaw(obj_id)
                else:
                    speed = self.k.vehicle.get_speed(obj_id)
                    yaw = self.k.vehicle.get_yaw(obj_id)
                num_obs = len(self.observation_names)
                observation[index * num_obs: (index + 1) * num_obs] = [rel_x, rel_y, speed, is_ped, yaw]
            obs.update({rl_id: observation})

        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        for rl_id in self.k.vehicle.get_rl_ids():

            # TODO(@evinitsky) pick the right reward
            reward = 0

            rewards[rl_id] = reward
        return rewards

    def find_visible_objects(self, veh_id, radius):
        """For a given vehicle ID, find the IDs of all the objects that are within a radius of them

        Parameters
        ----------
        veh_id : str
            The id of the vehicle whose visible objects we want to compute
        radius : float
            How large of a circle we want to search around

        Returns
        -------
        close_objects : [str]
            Returns a list of the IDs of pedestrians and cars that are within a radius of the car and are unobscured

        """
        # TODO(@nliu)
        pass
