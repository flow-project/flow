from cistar.envs.loop_accel import SimpleAccelerationEnvironment

from rllab.envs.base import Step

import numpy as np
import logging
import random
import traci

class PerturbationAccelerationLoop(SimpleAccelerationEnvironment):

    def __init__(self, env_params, sumo_binary, sumo_params, scenario):
        super().__init__(env_params, sumo_binary, sumo_params, scenario)

        if "perturbation_at" not in env_params:
            raise ValueError("Time for perturbation not specified")
        self.perturbation_at = env_params["perturbation_at"]

        if "perturbation_length" not in env_params:
            raise ValueError("Time for perturbation not specified")
        self.perturbation_length = env_params["perturbation_length"]

        if "perturbed_id" in env_params:
            self.perturbed_id = env_params["perturbed_id"]
        else:
            self.perturbed_id = list(self.vehicles.keys())[random.randint(0, len(self.vehicles.keys())-1)]

    def additional_command(self):
        if self.timer > self.perturbation_at and self.timer < (self.perturbation_at + self.perturbation_length):
            self.apply_action(self.perturbed_id, self.env_params["max-deacc"])
