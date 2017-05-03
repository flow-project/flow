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
        self.traci_connection.vehicle.setColor(self.perturbed_id, (0, 255, 255, 0))

    def additional_command(self):
        if self.num_perturbations < len(self.perturbations):
            if self.timer > self.perturbations[self.num_perturbations][0] \
                    and self.timer < (self.perturbations[self.num_perturbations][0]
                                          + self.perturbations[self.num_perturbations][1]):
                self.apply_action(self.perturbed_id, self.env_params["max-deacc"])
            if self.timer > (
                self.perturbations[self.num_perturbations][0] + self.perturbations[self.num_perturbations][1]):
                self.num_perturbations += 1
