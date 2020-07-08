import unittest
import os
import time
import csv

from flow.core.experiment import Experiment
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController, ContinuousRouter
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoParams
from flow.core.params import EnvParams, InitialConfig, NetParams
from flow.core.params import TrafficLightParams
from flow.envs import AccelEnv
from flow.networks import RingNetwork

from tests.setup_scripts import ring_road_exp_setup

import numpy as np

os.environ["TEST_FLAG"] = "True"


class TestNumSteps(unittest.TestCase):
    """
    Tests that experiment class runs for the number of steps requested.
    """

    def setUp(self):
        # create the environment and network classes for a ring road
        env, _, flow_params = ring_road_exp_setup()
        flow_params['sim'].render = False
        flow_params['env'].horizon = 10
        # instantiate an experiment class
        self.exp = Experiment(flow_params)
        self.exp.env = env

    def tearDown(self):
        # free up used memory
        self.exp = None

    def test_steps(self):
        self.exp.run(num_runs=1)

        self.assertEqual(self.exp.env.time_counter, 10)


class TestNumRuns(unittest.TestCase):
    """
    Tests that the experiment class properly resets as many times as requested,
    after the correct number of iterations.
    """

    def test_num_runs(self):
        # run the experiment for 1 run and collect the last position of all
        # vehicles
        env, _, flow_params = ring_road_exp_setup()
        flow_params['sim'].render = False
        flow_params['env'].horizon = 10
        exp = Experiment(flow_params)
        exp.env = env
        exp.run(num_runs=1)

        vel1 = [exp.env.k.vehicle.get_speed(exp.env.k.vehicle.get_ids())]

        # run the experiment for 2 runs and collect the last position of all
        # vehicles
        env, _, flow_params = ring_road_exp_setup()
        flow_params['sim'].render = False
        flow_params['env'].horizon = 10

        exp = Experiment(flow_params)
        exp.env = env
        exp.run(num_runs=2)

        vel2 = [exp.env.k.vehicle.get_speed(exp.env.k.vehicle.get_ids())]

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
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            num_vehicles=1)

        env, _, flow_params = ring_road_exp_setup(vehicles=vehicles)
        flow_params['sim'].render = False
        flow_params['env'].horizon = 10
        exp = Experiment(flow_params)
        exp.env = env
        exp.run(1, rl_actions=rl_actions)

        # check that the acceleration of the RL vehicle was that specified by
        # the rl_actions method
        self.assertAlmostEqual(exp.env.k.vehicle.get_speed("rl_0"), 1,
                               places=1)


class TestConvertToCSV(unittest.TestCase):
    """
    Tests that the emission files are converted to csv's if the parameter
    is requested.
    """

    def test_convert_to_csv(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sim_params = SumoParams(emission_path="{}/".format(dir_path))

        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            num_vehicles=1)

        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "sort_vehicles": False,
        }
        env_params = EnvParams(
            horizon=10,
            additional_params=additional_env_params)

        additional_net_params = {
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        flow_params = dict(
            exp_tag="RingRoadTest",
            env_name=AccelEnv,
            network=RingNetwork,
            simulator='traci',
            sim=sim_params,
            env=env_params,
            net=net_params,
            veh=vehicles,
            initial=InitialConfig(lanes_distribution=1),
            tls=TrafficLightParams(),
        )

        exp = Experiment(flow_params)
        exp.run(num_runs=1, convert_to_csv=True)

        time.sleep(1.0)

        # check that both the csv file exists and the xml file doesn't.
        self.assertFalse(os.path.isfile(dir_path + "/{}-0_emission.xml".format(
            exp.env.network.name)))
        self.assertTrue(os.path.isfile(dir_path + "/{}-0_emission.csv".format(
            exp.env.network.name)))

        # check that the keys within the emission file matches its expected
        # values
        with open(dir_path + "/{}-0_emission.csv".format(
                exp.env.network.name), "r") as f:
            reader = csv.reader(f)
            header = next(reader)

        self.assertListEqual(header, [
            "time",
            "id",
            "x",
            "y",
            "speed",
            "headway",
            "leader_id",
            "target_accel_with_noise_with_failsafe",
            "target_accel_no_noise_no_failsafe",
            "target_accel_with_noise_no_failsafe",
            "target_accel_no_noise_with_failsafe",
            "realized_accel",
            "road_grade",
            "edge_id",
            "lane_number",
            "distance",
            "relative_position",
            "follower_id",
            "leader_rel_speed",
        ])

        time.sleep(0.1)

        # delete the files
        os.remove(os.path.expanduser(dir_path + "/{}-0_emission.csv".format(
            exp.env.network.name)))


if __name__ == '__main__':
    unittest.main()
