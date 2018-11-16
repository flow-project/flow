import unittest
import os
from tests.setup_scripts import ring_road_exp_setup
from flow.core.params import EnvParams
from flow.core.vehicles import Vehicles
from flow.core.rewards import average_velocity, total_velocity, \
    desired_velocity, max_edge_velocity

os.environ["TEST_FLAG"] = "True"


class TestRewards(unittest.TestCase):
    """Tests for all methods in flow/core/rewards.py."""

    def test_desired_velocity(self):
        """Test the desired_velocity method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1})

        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params)

        # check that the fail attribute leads to a zero return
        self.assertEqual(desired_velocity(env, fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(desired_velocity(env, fail=False), 0)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 10)

        # check the new average speed
        self.assertAlmostEqual(desired_velocity(env, fail=False), 0.05131670)

    def test_average_velocity(self):
        """Test the average_velocity method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # check that the fail attribute leads to a zero return
        self.assertEqual(average_velocity(env, fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(average_velocity(env, fail=False), 0)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 10)

        # check the new average speed
        self.assertEqual(average_velocity(env, fail=False), 1)

        # recreate the environment with no vehicles
        vehicles = Vehicles()
        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # check that the reward function return 0 in the case of no vehicles
        self.assertEqual(average_velocity(env, fail=False), 0)

    def test_total_velocity(self):
        """Test the average_velocity method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # check that the fail attribute leads to a zero return
        self.assertEqual(total_velocity(env, fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(total_velocity(env, fail=False), 0)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 10)

        # check the new average speed
        self.assertEqual(total_velocity(env, fail=False), 10)

    def test_max_edge_velocity(self):
        """Test the average_velocity method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1})

        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params)

        # check that the fail attribute leads to a zero return
        self.assertEqual(max_edge_velocity(env, edge_list=[], fail=True), 0)

        # check the average speed upon reset
        self.assertEqual(max_edge_velocity(env, edge_list=["bottom"],
                                           fail=False), 0)

        # change the speed of one vehicle in the edge list
        env.vehicles.test_set_speed("test_0", 10)

        # check the new average speed
        self.assertAlmostEqual(max_edge_velocity(env, edge_list=["bottom"],
                                                 fail=False), 3.178372452)

        # change the speed of one of the vehicles outside the edge list
        env.vehicles.test_set_speed("test_8", 10)

        # check that the new average speed is the same as before
        self.assertAlmostEqual(max_edge_velocity(env, edge_list=["bottom"],
                                                 fail=False), 3.178372452)


if __name__ == '__main__':
    unittest.main()
