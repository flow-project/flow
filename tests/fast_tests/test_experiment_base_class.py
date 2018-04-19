import unittest
import os

from flow.core.experiment import SumoExperiment
from tests.setup_scripts import ring_road_exp_setup
import numpy as np

os.environ["TEST_FLAG"] = "True"


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

        vel1 = [exp.env.vehicles.get_speed(exp.env.vehicles.get_ids())]

        # run the experiment for 2 runs and collect the last position of all
        # vehicles
        env, scenario = ring_road_exp_setup()
        exp = SumoExperiment(env, scenario)
        exp.run(num_runs=2, num_steps=10)

        vel2 = [exp.env.vehicles.get_speed(exp.env.vehicles.get_ids())]

        # check that the final position is the same in both instances
        np.testing.assert_array_almost_equal(vel1, vel2)


if __name__ == '__main__':
    unittest.main()
