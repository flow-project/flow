import unittest
import logging
import numpy as np

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import *
from flow.controllers.rlcontroller import RLController

from flow.envs.loop_accel import SimpleAccelerationEnvironment
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario

from setup_scripts import ring_road_exp_setup


class TestCFMController(unittest.TestCase):
    """
    Tests that the CFM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"k_d": 1, "k_v": 1, "k_c": 1, "d_des": 1, "v_des": 8,
             "accel_max": 20, "decel_max": -5, "tau": 0, "dt": 0.1, "noise": 0}

        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(CFMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 15, 20, 25]
        test_speeds = [5, 10, 5, 10, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])
            self.env.vehicles.set_speed(veh_id, test_speeds[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [12, 2, 20, 12, 20]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestBCMController(unittest.TestCase):
    """
    Tests that the BCM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"k_d": 1, "k_v": 1, "k_c": 1, "d_des": 1, "v_des": 8,
             "accel_max": 15, "decel_max": -5, "tau": 0, "dt": 0.1, "noise": 0}

        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(BCMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 15, 20, 25]
        test_speeds = [5, 10, 5, 10, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])
            self.env.vehicles.set_speed(veh_id, test_speeds[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [-12, -7, 15, -7, 13]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestOVMController(unittest.TestCase):
    """
    Tests that the OVM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"alpha": 1, "beta": 1, "h_st": 2, "h_go": 15, "v_max": 30,
             "accel_max": 15, "decel_max": -5, "tau": 0, "dt": 0.1, "noise": 0}

        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(OVMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [0, 10, 5, 5, 5]
        test_speeds = [5, 10, 5, 10, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])
            self.env.vehicles.set_speed(veh_id, test_speeds[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [0, 5.319073, 3.772339, -5., -1.227661]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestLinearOVM(unittest.TestCase):
    """
    Tests that the Linear OVM Controller returning mathematically accurate
    values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"v_max": 30, "accel_max": 15, "decel_max": -5, "adaptation": 0.65,
             "h_st": 5, "tau": 0, "dt": 0.1, "noise": 0}

        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(LinearOVM, contr_params),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 10, 15, 0]
        test_speeds = [5, 10, 5, 10, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])
            self.env.vehicles.set_speed(veh_id, test_speeds[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [-5., -2.392308, 5.3, 10.6, -5.]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestIDMController(unittest.TestCase):
    """
    Tests that the IDM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = {"v0": 30, "T": 1, "a": 1, "b": 1.5, "delta": 4, "s0": 2,
                        "s1": 0, "decel_max": -5, "dt": 0.1, "noise": 0}

        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(IDMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [10, 20, 30, 40, 50]
        test_speeds = [5, 10, 5, 10, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])
            self.env.vehicles.set_speed(veh_id, test_speeds[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = \
            [0.959228, -1.638757,  0.994784,  0.331051,  0.979628]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)

        # set the perceived headway to zero
        test_headways = [0, 0, 0, 0, 0]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        # make sure the controller doesn't return a ZeroDivisionError when the
        # headway is zero
        [self.env.vehicles.get_acc_controller(veh_id).get_action(self.env)
         for veh_id in ids]


class TestInstantaneousFailsafe(unittest.TestCase):
    """
    Tests that the instantaneous failsafe of the base acceleration controller
    does not allow vehicles to crash under situations where they otherwise
    would.
    """
    def setUp_failsafe(self, vehicles):
        additional_env_params = {"target_velocity": 8, "max-deacc": 3,
                                 "max-acc": 3}
        env_params = EnvParams(additional_params=additional_env_params,
                               longitudinal_fail_safe="instantaneous")

        additional_net_params = {"length": 100, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=10)

        # create the environment and scenario classes for a ring road
        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params,
                                            net_params=net_params,
                                            initial_config=initial_config)

        # instantiate an experiment class
        self.exp = SumoExperiment(env, scenario)

    def tearDown_failsafe(self):
        # free data used by the class
        self.exp = None

    def test_no_crash_CFM(self):
        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(CFMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_BCM(self):
        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(BCMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_OVM(self):
        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(OVMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_LinearOVM(self):
        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(LinearOVM, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_IDM(self):
        vehicles = Vehicles()
        vehicles.add_vehicles(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()


class TestSafeVelocityFailsafe(TestInstantaneousFailsafe):
    """
    Tests that the safe velocity failsafe of the base acceleration controller
    does not fail under extreme conditions.
    """
    def setUp_failsafe(self, vehicles):
        additional_env_params = {"target_velocity": 8, "max-deacc": 3,
                                 "max-acc": 3}
        env_params = EnvParams(additional_params=additional_env_params,
                               longitudinal_fail_safe="safe_velocity")

        additional_net_params = {"length": 100, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=10)

        # create the environment and scenario classes for a ring road
        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params,
                                            net_params=net_params,
                                            initial_config=initial_config)

        # instantiate an experiment class
        self.exp = SumoExperiment(env, scenario)


class TestStaticLaneChanger(unittest.TestCase):
    """
    Makes sure that vehicles with a static lane-changing controller do not
    change lanes.
    """
    def setUp(self):
        # add an extra lane to the ring road network
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        ids = self.env.vehicles.get_ids()

        # run the experiment for a few iterations and collect the lane index
        # for vehicles
        lanes = [self.env.vehicles.get_lane(veh_id) for veh_id in ids]
        for i in range(5):
            self.env._step(rl_actions=[])
            lanes += [self.env.vehicles.get_lane(veh_id) for veh_id in ids]

        # set the timer as very high and reset (the timer used to cause bugs at
        # the beginning of a new run for this controller)
        self.env.timer = 10000
        self.env.reset()

        # run the experiment for a few more iterations and collect the lane
        # index for vehicles
        lanes = [self.env.vehicles.get_lane(veh_id) for veh_id in ids]
        for i in range(5):
            self.env._step(rl_actions=[])
            lanes += [self.env.vehicles.get_lane(veh_id) for veh_id in ids]

        # assert that all lane indices are zero
        self.assertEqual(sum(np.array(lanes)), 0)


class TestContinuousRouter(unittest.TestCase):
    """
    Tests that the continuous router operates properly if there is no need to
    reroute, and if there is a need to do so.
    """
    def setUp(self):
        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup()

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        veh_id = self.env.vehicles.get_ids()[0]

        # set the perceived route of the vehicle
        self.env.vehicles.set_route(veh_id, ["bottom", "right", "top", "left"])

        # set the perceived edge of the vehicle at the beginning of its route
        self.env.vehicles.set_edge(veh_id, "bottom")

        # assert that the controller is returning a None value
        requested_route = self.env.vehicles.get_routing_controller(
            veh_id).choose_route(self.env)

        self.assertIsNone(requested_route)

        # set the perceived edge of the vehicle at the middle of its route
        self.env.vehicles.set_edge(veh_id, "right")

        # assert that the controller is returning a None value
        requested_route = self.env.vehicles.get_routing_controller(
            veh_id).choose_route(self.env)

        self.assertIsNone(requested_route)

        # set the perceived edge of the vehicle at the end of its route
        self.env.vehicles.set_edge(veh_id, "left")

        # assert that the controller is returning a list of edges starting at
        # this link and then containing the route of the link ahead of it
        requested_route = self.env.vehicles.get_routing_controller(
            veh_id).choose_route(self.env)

        expected_route = ["left", "bottom", "right", "top"]

        self.assertSequenceEqual(requested_route, expected_route)


if __name__ == '__main__':
    unittest.main()
