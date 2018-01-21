from flow.envs.loop_accel import AccelEnv

import numpy as np
import random


class PerturbationAccelerationLoop(AccelEnv):

    def __init__(self, env_params, sumo_params, scenario):
        super().__init__(env_params, sumo_params, scenario)

        if "perturbations" not in env_params:
            raise ValueError("Perturbation not specified")
        self.perturbations = env_params.get_additional_param("perturbations")
        self.num_perturbations = 0
        # self.perturbation_at = env_params["perturbation_at"]

        # if "perturbation_length" not in env_params:
        #     raise ValueError("Time for perturbation not specified")
        # self.perturbation_length = env_params["perturbation_length"]

        if "perturbed_id" in env_params:
            self.perturbed_id = env_params.get_additional_param("perturbed_id")
        else:
            ids = self.vehicles.get_ids()
            self.perturbed_id = ids[random.randint(0, len(ids)-1)]
        self.traci_connection.vehicle.setColor(self.perturbed_id, (0, 255, 255, 0))

    def additional_command(self):
        if self.num_perturbations < len(self.perturbations):
            if self.perturbations[self.num_perturbations][0] < self.time_counter < \
                    (self.perturbations[self.num_perturbations][0]
                     + self.perturbations[self.num_perturbations][1]):
                self.apply_acceleration(
                    self.perturbed_id,
                    self.env_params.max_deacc)
            if self.time_counter > (
                self.perturbations[self.num_perturbations][0] +
                    self.perturbations[self.num_perturbations][1]):
                self.num_perturbations += 1
