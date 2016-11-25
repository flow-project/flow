import numpy as np
# import matplotlib as plt
from rllab.envs.base import Env
from rllab.spaces import Box
# from rllab.spaces import Product
from rllab.envs.base import Step

class TrafficEnv(Env):
    """
    Toy model with 1 automated car on 1 lane highway
    Controls are just the velocity the car should go at
    # """
    MAX_VELOCITY = 45
    # delta = 1
    # max_acceleration = 1.5
    # min_acceleration = -4.5

    def __init__(self):
        Env.__init__(self)
        self.num_cars = 2

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._state = self._state + action
        reward = self.compute_reward(self._state)
        done = np.all(abs(self._state-self.MAX_VELOCITY) < 0.01)
        # done = np.all(self._state > (self.MAX_VELOCITY - self.delta)) #and self._state < (self.MAX_VELOCITY + self.delta)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def compute_reward(self, velocity):
        return -np.linalg.norm(velocity - self.MAX_VELOCITY)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self state is velocity, observation is velocity
        # self._state = np.random.uniform(0, self.MAX_VELOCITY, size=(self.num_cars,))
        self._state = np.ones(self.num_cars,)*42
        observation = np.copy(self._state)
        return observation

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return Box(low=-5, high=5, shape=(self.num_cars, ))

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(low=-np.inf, high=np.inf, shape=(self.num_cars,))
        # pos = Box(low=0., high=self.road_length, shape=(self.num_cars, ))
        # vel = Box(low=0., high=self.MAX_VELOCITY+10, shape=(self.num_cars, ))
        # accel = Box(low=self.min_acceleration, high=self.max_acceleration, shape=(self.num_cars, ))
        # return vel
        # return Product([pos, vel, accel])

    def render(self):
        print('current state/velocity:', self._state)