import unittest
import os

from flow.core.experiment import SumoExperiment
from flow.core.vehicles import Vehicles
from flow.controllers import RLController, ContinuousRouter
from flow.core.params import SumoCarFollowingParams

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

    def test_steps(self):
        self.exp.run(num_runs=1, num_steps=10)

        self.assertEqual(self.exp.env.time_counter, 10)


class TestNumRuns(unittest.TestCase):
    """
    Tests that the experiment class properly resets as many times as requested,
    after the correct number of iterations.
    """

    def test_num_runs(self):
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


class TestRLActions(unittest.TestCase):
    """
    Test that the rl_actions parameter acts as it should when it is specified,
    and does not break the simulation when it is left blank.
    """

    def test_rl_actions(self):
        def rl_actions(*_):
            return [1]  # actions are always an acceleration of 1 for one veh

        # create an environment using AccelEnv with 1 RL vehicle
        vehicles = Vehicles()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            num_vehicles=1)

        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        exp = SumoExperiment(env=env, scenario=scenario)

        exp.run(1, 10, rl_actions=rl_actions)

        # check that the acceleration of the RL vehicle was that specified by
        # the rl_actions method
        self.assertAlmostEqual(exp.env.vehicles.get_speed("rl_0"), 1, places=1)

        pass


if __name__ == '__main__':
    unittest.main()
