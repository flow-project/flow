import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, \
    NetParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.controllers import RLController
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS
from flow.utils.exceptions import FatalFlowError
from flow.envs import Env, TestEnv

from tests.setup_scripts import ring_road_exp_setup, highway_exp_setup
import os
import numpy as np

os.environ["TEST_FLAG"] = "True"

# colors for vehicles
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class TestShuffle(unittest.TestCase):
    """
    Tests that, at resets, the ordering of vehicles changes while the starting
    position values stay the same.
    """

    def setUp(self):
        # turn on vehicle arrangement shuffle
        env_params = EnvParams(
            additional_params=ADDITIONAL_ENV_PARAMS)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5)

        initial_config = InitialConfig(x0=5, shuffle=True)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(
            env_params=env_params,
            initial_config=initial_config,
            vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_shuffle(self):
        ids = self.env.k.vehicle.get_ids()

        # position of vehicles before reset
        before_reset = [self.env.k.vehicle.get_x_by_id(veh_id)
                        for veh_id in ids]

        # reset the environment
        self.env.reset()

        # position of vehicles after reset
        after_reset = [self.env.k.vehicle.get_x_by_id(veh_id)
                       for veh_id in ids]

        self.assertCountEqual(before_reset, after_reset)


class TestEmissionPath(unittest.TestCase):
    """
    Tests that the default emission path of an environment is set to None.
    If it is not None, then sumo starts accumulating memory.
    """

    def setUp(self):
        # set sim_params to default
        sim_params = SumoParams()

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(sim_params=sim_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_emission(self):
        self.assertIsNone(self.env.sim_params.emission_path)


class TestApplyingActionsWithSumo(unittest.TestCase):
    """
    Tests the apply_acceleration, apply_lane_change, and choose_routes
    functions in base_env.py
    """

    def setUp(self):
        # create a 2-lane ring road network
        additional_net_params = {
            "length": 230,
            "lanes": 3,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # turn on starting position shuffle
        env_params = EnvParams(
            additional_params=ADDITIONAL_ENV_PARAMS)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                accel=1000, decel=1000),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0),
            num_vehicles=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(
            net_params=net_params, env_params=env_params, vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_apply_acceleration(self):
        """
        Tests that, in the absence of all failsafes, the acceleration requested
        from sumo is equal to the acceleration witnessed in between steps. Also
        ensures that vehicles can never have velocities below zero given any
        acceleration.
        """
        ids = self.env.k.vehicle.get_ids()

        vel0 = np.array(
            [self.env.k.vehicle.get_speed(veh_id) for veh_id in ids])

        # apply a certain set of accelerations to the vehicles in the network
        accel_step0 = np.array([0, 1, 4, 9, 16])
        self.env.k.vehicle.apply_acceleration(veh_ids=ids, acc=accel_step0)
        self.env.k.simulation.simulation_step()
        self.env.k.vehicle.update(False)

        # compare the new velocity of the vehicles to the expected velocity
        # given the accelerations
        vel1 = np.array([
            self.env.k.vehicle.get_speed(veh_id)
            for veh_id in ids
        ])
        expected_vel1 = (vel0 + accel_step0 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel1, expected_vel1, 1)

        # apply a set of decelerations
        accel_step1 = np.array([-16, -9, -4, -1, 0])
        self.env.k.vehicle.apply_acceleration(veh_ids=ids, acc=accel_step1)
        self.env.k.simulation.simulation_step()
        self.env.k.vehicle.update(False)

        # this time, some vehicles should be at 0 velocity (NOT less), and sum
        # are a result of the accelerations that took place
        vel2 = np.array([
            self.env.k.vehicle.get_speed(veh_id)
            for veh_id in ids
        ])
        expected_vel2 = (vel1 + accel_step1 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel2, expected_vel2, 1)

    def test_apply_lane_change_errors(self):
        """
        Ensures that apply_lane_change raises ValueErrors when it should
        """
        self.env.reset()
        ids = self.env.k.vehicle.get_ids()

        # make sure that running apply lane change with a invalid direction
        # values leads to a ValueError
        bad_directions = np.array([-1, 0, 1, 2, 3])

        self.assertRaises(
            ValueError,
            self.env.k.vehicle.apply_lane_change,
            veh_ids=ids,
            direction=bad_directions)

    def test_apply_lane_change_direction(self):
        """
        Tests the direction method for apply_lane_change. Ensures that the lane
        change action requested from sumo is the same as the lane change that
        occurs, and that vehicles attempting do not issue lane changes in there
        is no lane in te requested direction.
        """
        self.env.reset()
        ids = self.env.k.vehicle.get_ids()
        lane0 = np.array(
            [self.env.k.vehicle.get_lane(veh_id) for veh_id in ids])
        max_lanes = self.env.net_params.additional_params['lanes']

        # perform lane-changing actions using the direction method
        direction0 = np.array([0, 1, 0, 1, -1])
        self.env.k.vehicle.apply_lane_change(ids, direction=direction0)
        self.env.k.simulation.simulation_step()
        self.env.k.vehicle.update(False)

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane1 = np.array([
            self.env.k.vehicle.get_lane(veh_id)
            for veh_id in ids
        ])
        expected_lane1 = (lane0 + np.sign(direction0)).clip(
            min=0, max=max_lanes - 1)

        np.testing.assert_array_almost_equal(lane1, expected_lane1, 1)

        # perform lane-changing actions using the direction method one more
        # time to test lane changes to the right
        direction1 = np.array([-1, -1, -1, -1, -1])
        self.env.k.vehicle.apply_lane_change(ids, direction=direction1)
        self.env.k.simulation.simulation_step()
        self.env.k.vehicle.update(False)

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane2 = np.array([
            self.env.k.vehicle.get_lane(veh_id)
            for veh_id in ids
        ])
        expected_lane2 = (lane1 + np.sign(direction1)).clip(
            min=0, max=max_lanes - 1)

        np.testing.assert_array_almost_equal(lane2, expected_lane2, 1)


class TestWarmUpSteps(unittest.TestCase):
    """Ensures that the appropriate number of warmup steps are run when using
    flow.core.params.EnvParams.warmup_steps"""

    def test_it_works(self):
        warmup_step = 5  # some value

        # start an environment with a number of simulations per step greater
        # than one
        env_params = EnvParams(
            warmup_steps=warmup_step, additional_params=ADDITIONAL_ENV_PARAMS)
        env, scenario = ring_road_exp_setup(env_params=env_params)

        # time before running a reset
        t1 = 0
        # perform a reset
        env.reset()
        # time after a reset
        t2 = env.time_counter

        # ensure that the difference in time is equal to sims_per_step
        self.assertEqual(t2 - t1, warmup_step)


class TestSimsPerStep(unittest.TestCase):
    """Ensures that the appropriate number of simultaions are run at any given
    steps when using flow.core.params.EnvParams.sims_per_step"""

    def test_it_works(self):
        sims_per_step = 5  # some value

        # start an environment with a number of simulations per step greater
        # than one
        env_params = EnvParams(
            sims_per_step=sims_per_step,
            additional_params=ADDITIONAL_ENV_PARAMS)
        env, scenario = ring_road_exp_setup(env_params=env_params)

        env.reset()
        # time before running a step
        t1 = env.time_counter
        # perform a step
        env.step(rl_actions=[])
        # time after a step
        t2 = env.time_counter

        # ensure that the difference in time is equal to sims_per_step
        self.assertEqual(t2 - t1, sims_per_step)


class TestAbstractMethods(unittest.TestCase):
    """
    These series of tests are meant to ensure that the environment abstractions
    exist and are in fact abstract, i.e. they will raise errors if not
    implemented in a child class.
    """

    def setUp(self):
        env, scenario = ring_road_exp_setup()
        sim_params = SumoParams()  # FIXME: make ambiguous
        env_params = EnvParams()
        self.env = Env(sim_params=sim_params,
                       env_params=env_params,
                       scenario=scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_get_state(self):
        """Checks that get_state raises an error."""
        self.assertRaises(NotImplementedError, self.env.get_state)

    def test_compute_reward(self):
        """Checks that compute_reward returns 0."""
        self.assertEqual(self.env.compute_reward([]), 0)

    def test__apply_rl_actions(self):
        self.assertRaises(NotImplementedError, self.env._apply_rl_actions,
                          rl_actions=None)


class TestVehicleColoring(unittest.TestCase):

    def test_all(self):
        vehicles = VehicleParams()
        vehicles.add("human", num_vehicles=10)
        # add an RL vehicle to ensure that its color will be distinct
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        _, scenario = ring_road_exp_setup(vehicles=vehicles)
        env = TestEnv(EnvParams(), SumoParams(), scenario)
        env.reset()

        # set one vehicle as observed
        env.k.vehicle.set_observed("human_0")

        # update the colors of all vehicles
        env.step(rl_actions=None)

        # check that, when rendering is off, the colors don't change (this
        # avoids unnecessary API calls)
        for veh_id in env.k.vehicle.get_ids():
            self.assertEqual(env.k.vehicle.get_color(veh_id), YELLOW)

        # a little hack to ensure the colors change
        env.sim_params.render = True

        # set one vehicle as observed
        env.k.vehicle.set_observed("human_0")

        # update the colors of all vehicles
        env.step(rl_actions=None)

        # check the colors of all vehicles
        for veh_id in env.k.vehicle.get_ids():
            if veh_id in ["human_0"]:
                self.assertEqual(env.k.vehicle.get_color(veh_id), CYAN)
            elif veh_id == "rl_0":
                self.assertEqual(env.k.vehicle.get_color(veh_id), RED)
            else:
                self.assertEqual(env.k.vehicle.get_color(veh_id), WHITE)


class TestNotEnoughVehicles(unittest.TestCase):
    """Tests that when not enough vehicles spawn an error is raised."""

    def test_num_spawned(self):
        initial_config = InitialConfig(
            spacing="custom",
            additional_params={
                'start_positions': [('highway_0', 0), ('highway_0', 0)],
                'start_lanes': [0, 0]}
        )
        vehicles = VehicleParams()
        vehicles.add('test', num_vehicles=2)

        self.assertRaises(FatalFlowError,
                          highway_exp_setup,
                          initial_config=initial_config,
                          vehicles=vehicles)


if __name__ == '__main__':
    unittest.main()
