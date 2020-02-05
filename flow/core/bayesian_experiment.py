"""Contains an experiment class for running bayesian simulations."""

import logging
import datetime
import numpy as np
import time
import os

from flow.core.experiment import Experiment
from flow.core.util import emission_to_csv


class BayesianExperiment(Experiment):

    def __init__(self, env):
        super().__init__(env)

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False, collect_data=True):
        """
        See parent class
        """

        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info("Iter #" + str(i))
            ret = 0
            ret_list = []
            state = self.env.reset()
            for j in range(num_steps):

                state, reward, done, _ = self.env.step(rl_actions(state))
        return

