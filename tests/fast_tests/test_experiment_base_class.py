import unittest
import os
os.environ["TEST_FLAG"] = "True"

from flow.core.experiment import SumoExperiment
from tests.setup_scripts import ring_road_exp_setup
import numpy as np


class TestNumSteps(unittest.TestCase):
    """
    Tests that experiment class runs for the number of steps requested.
    """

    def setUp(self):
        # create the environment and scenario classes for a ring road
        env, scenario = ring_road_exp_setup()

        # instantiate an experiment class
        self.exp = SumoExperiment(env, scenario)

    def tearDown(self):
        # free up used memory
        self.exp = None

    def runTest(self):
        self.exp.run(num_runs=1, num_steps=10)

        self.assertEqual(self.exp.env.time_counter, 10)


class TestNumRuns(unittest.TestCase):
    """
    Tests that the experiment class properly resets as many times as requested,
    after the correct number of iterations.
    """

    def runTest(self):
        # run the experiment for 1 run and collect the last position of all
        # vehicles
        env, scenario = ring_road_exp_setup()
        exp = SumoExperiment(env, scenario)
        exp.run(num_runs=1, num_steps=10)

        pos1 = [exp.env.vehicles.get_speed()]

        # run the experiment for 2 runs and collect the last position of all
        # vehicles
        env, scenario = ring_road_exp_setup()
        exp = SumoExperiment(env, scenario)
        exp.run(num_runs=2, num_steps=10)

        pos2 = [exp.env.vehicles.get_speed()]

        # check that the final position is the same in both instances
        np.testing.assert_array_almost_equal(pos1, pos2)


if __name__ == '__main__':
    unittest.main()
