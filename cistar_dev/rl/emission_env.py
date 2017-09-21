import numpy as np
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.spaces import Product
from rllab.envs.base import Step

from scripts.sumo_config import *
from sumolib import checkBinary

import subprocess, sys
import traci
import traci.constants as tc

class EmissionEnv(Env):
    """
    Toy model, number of cars determined by SUMO config files
    Controls are just the velocity the car should go at
    # """
    GOAL_VELOCITY = 25 # 45
    delta = 0.01
    # max_acceleration = 1.5
    # min_acceleration = -4.5
    # PORT = defaults.PORT
    PORT = PORT
    sumoBinary = checkBinary(BINARY)

    def __init__(self, num_cars, total_cars, cfgfn, highway_length, fullbool=True, expandedstate=True):
        Env.__init__(self)
        self.num_cars = num_cars
        self.tot_cars = total_cars
        self.car_ids = [str(i) for i in range(total_cars)]
        # This works best if num_cars is evenly divisible to total_cars
        self.controllable = self.find_controllable(self.car_ids, self.num_cars)
        self.ctrl_ids = [int(i) for i in self.controllable]
        self.cfgfn = cfgfn
        self.initialized = False

        self.highway_length = highway_length
        edgelen = highway_length/4.
        # self.edgeorder = ["left", "top", "right", "bottom"]
        self.lanestarts = {"left": 3 * edgelen,
                       "top": 2*edgelen,
                       "right": edgelen,
                       "bottom": 0}

        self.fullbool = fullbool
        self.expandedstate = expandedstate

    def find_controllable(self, lst, ctrl):
        lst = np.array(lst)
        eff_a = len(lst) - (len(lst) % ctrl)
        idx = np.arange(0, eff_a, int(eff_a/ctrl))
        return lst[idx]

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
        ctrl_ids = self.ctrl_ids if self.fullbool else range(self.num_cars)
        new_speed = self._state[0, ctrl_ids] + action # self.ctrl_ids
        new_speed[np.where(new_speed < 0)] = 0
        for car_idx in range(self.num_cars):
            # almost instantaneous
            traci.vehicle.slowDown(self.controllable[car_idx], new_speed[car_idx], 1)
        traci.simulationStep()
        
        ctrl_ids = self.car_ids if self.fullbool else self.controllable
        if self.expandedstate:
            self._state = np.array([[traci.vehicle.getSpeed(vID), \
                                    self.get_lane_position(vID), \
                                    traci.vehicle.getLaneIndex(vID)] for vID in ctrl_ids]).T
        else:
            self._state = np.array([[traci.vehicle.getSpeed(vID) for vID in ctrl_ids]])

        
        # print("IN STEP, STATE", self._state.shape)
        # Horizontal concat of vertical vectors
        # self._state = np.concatenate((velocities.reshape(len(velocities), 1).T, ), axis=1)
        # done = np.all(abs(self._state-self.GOAL_VELOCITY) < self.delta)
        next_observation = np.copy(self._state)

        reward = self.compute_reward()
        return Step(observation=next_observation, reward=reward, done=False)

    def get_lane_position(self, vID):
        lanepos = traci.vehicle.getLanePosition(vID)
        lane = traci.vehicle.getLaneID(vID).split("_")
        return self.lanestarts[lane[0]] + lanepos

    def compute_reward(self):
        # Neg return on global (sum) fuel consumption
        return -np.sum([traci.vehicle.getFuelConsumption(carID) for carID in self.car_ids])

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        # self state is velocity, observation is velocity
        if self.initialized:
            traci.close()
        
        sumoProcess = subprocess.Popen([self.sumoBinary, "-c", self.cfgfn, "--remote-port", str(self.PORT)], stdout=sys.stdout, stderr=sys.stderr)
        traci.init(self.PORT)
        traci.simulationStep()
        
        if not self.initialized:
            self.vehIDs = traci.vehicle.getIDList()
            # print("ID List in reset", self.vehIDs)
            self.initialized = True
        
        ctrl_ids = self.car_ids if self.fullbool else self.controllable
        if self.expandedstate:
            self._state = np.array([[traci.vehicle.getSpeed(vID), \
                                    self.get_lane_position(vID), \
                                    traci.vehicle.getLaneIndex(vID)] for vID in ctrl_ids]).T
        else:
            self._state = np.array([[traci.vehicle.getSpeed(vID) for vID in ctrl_ids]])

        # print("IN RESET, STATE", self._state.shape)

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
        num_cars = self.tot_cars if self.fullbool else self.num_cars
        ypos = Box(low=0., high=np.inf, shape=(num_cars, ))
        vel = Box(low=0., high=np.inf, shape=(num_cars, ))
        xpos = Box(low=0., high=2., shape=(num_cars, ))
        return Product([vel, ypos, xpos]) if self.expandedstate else vel

    def render(self):
        print('current state/velocity, ypos, xpos:', self._state)

    def close(self):
        traci.close()
