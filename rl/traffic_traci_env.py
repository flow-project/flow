import numpy as np
# import matplotlib as plt
from rllab.envs.base import Env
from rllab.spaces import Box
# from rllab.spaces import Product
from rllab.envs.base import Step

import sumo_config as defaults
from sumolib import checkBinary

import subprocess, sys
import traci
import traci.constants as tc

class TrafficEnv(Env):
    """
    Toy model with 1 automated car on 1 lane highway
    Controls are just the velocity the car should go at
    # """
    GOAL_VELOCITY = 25 # 45
    delta = 0.01
    # max_acceleration = 1.5
    # min_acceleration = -4.5
    PORT = defaults.PORT
    sumoBinary = checkBinary(defaults.BINARY)

    def __init__(self):
        Env.__init__(self)
        self.num_cars = 2
        self.controllable = ['0', '11']
        self.cfgfn = "traffic/mit_circle/circular300.sumo.cfg"
        # This has now been moved into the reset function
        # sumoProcess = subprocess.Popen([self.sumoBinary, "-c", self.cfgfn, "--remote-port", str(self.PORT)], stdout=sys.stdout, stderr=sys.stderr)
        # traci.init(self.PORT)
        # traci.simulationStep()
        # self.vehIDs = traci.vehicle.getIDList()
        self.initialized = False
        # print("ID List", self.vehIDs)

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
        new_speed = self._state + action
        new_speed[np.where(new_speed < 0)] = 0
        for car_idx in range(self.num_cars):
            # almost instantaneous
            traci.vehicle.slowDown(self.controllable[car_idx], new_speed[car_idx], 1)
        traci.simulationStep()
        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.controllable])
        reward = self.compute_reward(self._state)
        # done = np.all(abs(self._state-self.GOAL_VELOCITY) < self.delta)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=False)

    def compute_reward(self, velocity):
        return -np.linalg.norm(velocity - self.GOAL_VELOCITY)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self state is velocity, observation is velocity
        # self._state = np.random.uniform(0, self.GOAL_VELOCITY, size=(self.num_cars,))
        # print("In reset function", self.initialized)
        if self.initialized:
            traci.close()
        sumoProcess = subprocess.Popen([self.sumoBinary, "-c", self.cfgfn, "--remote-port", str(self.PORT)], stdout=sys.stdout, stderr=sys.stderr)
        traci.init(self.PORT)
        traci.simulationStep()
        if not self.initialized:
            self.vehIDs = traci.vehicle.getIDList()
            print("ID List in reset", self.vehIDs)
            self.initialized = True
        self._state = np.array([traci.vehicle.getSpeed(vID) for vID in self.controllable])
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
        # vel = Box(low=0., high=self.GOAL_VELOCITY+10, shape=(self.num_cars, ))
        # accel = Box(low=self.min_acceleration, high=self.max_acceleration, shape=(self.num_cars, ))
        # return vel
        # return Product([pos, vel, accel])

    def render(self):
        print('current state/velocity:', self._state)

    def close(self):
        traci.close()
