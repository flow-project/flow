import unittest
import os
import numpy as np
from tests.setup_scripts import ring_road_exp_setup
from flow.core.params import EnvParams
from flow.core.params import VehicleParams
from flow.core.rewards import average_velocity, min_delay
from flow.core.rewards import desired_velocity, boolean_action_penalty
from flow.core.rewards import penalize_near_standstill, penalize_standstill
from flow.core.rewards import energy_consumption

os.environ["TEST_FLAG"] = "True"


class TestRewards(unittest.TestCase):
    """Tests for all methods in flow/core/rewards.py."""

    def test_desired_velocity(self):
        """Test the desired_velocity method."""
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": np.sqrt(10), "max_accel": 1, "max_decel": 1,
            "sort_vehicles": False})

        env, _, _ = ring_road_exp_setup(vehicles=vehicles,
                                        env_params=env_params)

        # check that the fail attribute leads to a zero return
        self.assertEqual(desired_velocity(env, fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(desired_velocity(env, fail=False), 0)

        # check the average speed upon reset with a subset of edges
        self.assertEqual(desired_velocity(env, edge_list=["bottom"],
                                          fail=False), 0)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", np.sqrt(10))

        # check the new average speed
        self.assertAlmostEqual(desired_velocity(env, fail=False),
                               1 - np.sqrt(90) / 10)

        # check the new average speed for a subset of edges
        self.assertAlmostEqual(desired_velocity(env, edge_list=["bottom"],
                                                fail=False),
                               1 - np.sqrt(20) / np.sqrt(30))

        # change the speed of one of the vehicles outside the edge list
        env.k.vehicle.test_set_speed("test_8", 10)

        # check that the new average speed is the same as before
        self.assertAlmostEqual(desired_velocity(env, edge_list=["bottom"],
                                                fail=False),
                               1 - np.sqrt(20) / np.sqrt(30))

    def test_average_velocity(self):
        """Test the average_velocity method."""
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)

        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # check that the fail attribute leads to a zero return
        self.assertEqual(average_velocity(env, fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(average_velocity(env, fail=False), 0)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", 10)

        # check the new average speed
        self.assertEqual(average_velocity(env, fail=False), 1)

        # recreate the environment with no vehicles
        vehicles = VehicleParams()
        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # check that the reward function return 0 in the case of no vehicles
        self.assertEqual(average_velocity(env, fail=False), 0)

    def test_min_delay(self):
        """Test the min_delay method."""
        # try the case of an environment with no vehicles
        vehicles = VehicleParams()
        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # check that the reward function return 0 in the case of no vehicles
        self.assertEqual(min_delay(env), 0)

        # try the case of multiple vehicles
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)
        env, _, _ = ring_road_exp_setup(vehicles=vehicles)

        # check the min_delay upon reset
        self.assertAlmostEqual(min_delay(env), 0)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", 10)

        # check the min_delay with the new speed
        self.assertAlmostEqual(min_delay(env), 0.0333333333333)

    def test_penalize_standstill(self):
        """Test the penalize_standstill method."""
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1,
            "sort_vehicles": False})

        env, _, _ = ring_road_exp_setup(vehicles=vehicles,
                                        env_params=env_params)

        # check the penalty is acknowledging all vehicles
        self.assertEqual(penalize_standstill(env, gain=1), -10)
        self.assertEqual(penalize_standstill(env, gain=2), -20)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", 10)

        # check the penalty is acknowledging all vehicles but one
        self.assertEqual(penalize_standstill(env, gain=1), -9)
        self.assertEqual(penalize_standstill(env, gain=2), -18)

    def test_penalize_near_standstill(self):
        """Test the penalize_near_standstill method."""
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1,
            "sort_vehicles": False})

        env, _, _ = ring_road_exp_setup(vehicles=vehicles,
                                        env_params=env_params)

        # check the penalty is acknowledging all vehicles
        self.assertEqual(penalize_near_standstill(env, gain=1), -10)
        self.assertEqual(penalize_near_standstill(env, gain=2), -20)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", 1)

        # check the penalty with good and bad thresholds
        self.assertEqual(penalize_near_standstill(env, thresh=2), -10)
        self.assertEqual(penalize_near_standstill(env, thresh=0.5), -9)

    def test_energy_consumption(self):
        """Test the energy consumption method."""
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1,
            "sort_vehicles": False})

        env, _, _ = ring_road_exp_setup(vehicles=vehicles,
                                        env_params=env_params)

        # check the penalty is zero at speed zero
        self.assertEqual(energy_consumption(env, gain=1), 0)

        # change the speed of one vehicle
        env.k.vehicle.test_set_speed("test_0", 1)
        self.assertEqual(energy_consumption(env), -12.059337750000001)

        # check that stepping change the previous speeds and increases the energy consumption
        env.step(rl_actions=None)
        env.step(rl_actions=None)
        self.assertGreater(env.k.vehicle.get_previous_speed("test_0"), 0.0)
        self.assertLess(energy_consumption(env), -12.059337750000001)

    def test_boolean_action_penalty(self):
        """Test the boolean_action_penalty method."""
        actions = [False, False, False, False, False]
        self.assertEqual(boolean_action_penalty(actions, gain=1), 0)
        self.assertEqual(boolean_action_penalty(actions, gain=2), 0)

        actions = [True, False, False, False, False]
        self.assertEqual(boolean_action_penalty(actions, gain=1), 1)
        self.assertEqual(boolean_action_penalty(actions, gain=2), 2)

        actions = [True, False, False, True, False]
        self.assertEqual(boolean_action_penalty(actions, gain=1), 2)
        self.assertEqual(boolean_action_penalty(actions, gain=2), 4)


if __name__ == '__main__':
    unittest.main()
