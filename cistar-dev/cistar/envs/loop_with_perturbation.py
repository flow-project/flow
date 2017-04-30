from cistar.envs.loop_accel import SimpleAccelerationEnvironment

from rllab.envs.base import Step

import numpy as np
import logging
import random
import traci

class PerturbationAccelerationLoop(SimpleAccelerationEnvironment):

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        if "perturbations" not in env_params:
            raise ValueError("Perturbation not specified")
        self.perturbations = env_params["perturbations"]
        self.num_perturbations = 0
        # self.perturbation_at = env_params["perturbation_at"]

        # if "perturbation_length" not in env_params:
        #     raise ValueError("Time for perturbation not specified")
        # self.perturbation_length = env_params["perturbation_length"]

        if "perturbed_id" in env_params:
            self.perturbed_id = env_params["perturbed_id"]
        else:
            self.perturbed_id = list(self.vehicles.keys())[random.randint(0, len(self.vehicles.keys())-1)]
        traci.vehicle.setColor(self.perturbed_id, (0, 255, 255, 0))

    def step(self, rl_actions):
        """
        Run one timestep of the environment's dynamics. "Self-driving cars" will
        step forward based on rl_actions, provided by the RL algorithm. Other cars
        will step forward based on their car following model. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        rl_actions : an action provided by the rl algorithm
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        logging.debug("================= performing step =================")
        for veh_id in self.controlled_ids:
            action = self.vehicles[veh_id]['controller'].get_action(self)
            safe_action = self.vehicles[veh_id]['controller'].get_safe_action(self, action)
            self.apply_action(veh_id, action=safe_action)
            logging.debug("Car with id " + veh_id + " is on route " + str(traci.vehicle.getRouteID(veh_id)))

        for index, veh_id in enumerate(self.rl_ids):
            action = rl_actions[index]
            safe_action = self.vehicles[veh_id]['controller'].get_safe_action(self, action)
            self.apply_action(veh_id, action=safe_action)

        self.timer += 1
        # TODO: Turn 100 into a hyperparameter
        # if it's been long enough try and change lanes
        if self.timer % 100 == 0:
            for veh_id in self.controlled_ids:
                newlane = self.vehicles[veh_id]['lane_changer'].get_action(self)
                traci.vehicle.changeLane(veh_id, newlane, 10000)

        if self.num_perturbations < len(self.perturbations):
            if self.timer > self.perturbations[self.num_perturbations][0] \
                and self.timer < (self.perturbations[self.num_perturbations][0] + self.perturbations[self.num_perturbations][1]):
                self.apply_action(self.perturbed_id, self.env_params["max-deacc"])
            if self.timer > (self.perturbations[self.num_perturbations][0] + self.perturbations[self.num_perturbations][1]):
                self.num_perturbations += 1

        traci.simulationStep()

        for veh_id in self.ids:
            self.vehicles[veh_id]["type"] = traci.vehicle.getTypeID(veh_id)
            self.vehicles[veh_id]["edge"] = traci.vehicle.getRoadID(veh_id)
            self.vehicles[veh_id]["position"] = traci.vehicle.getLanePosition(veh_id)
            self.vehicles[veh_id]["lane"] = traci.vehicle.getLaneIndex(veh_id)
            self.vehicles[veh_id]["speed"] = traci.vehicle.getSpeed(veh_id)

        # TODO: Can self._state be initialized, saved and updated so that we can
        # exploit numpy speed
        self._state = self.getState()
        reward = self.compute_reward(self._state)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=False)

