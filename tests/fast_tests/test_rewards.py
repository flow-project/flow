import unittest
import os
from tests.setup_scripts import ring_road_exp_setup
from flow.core.params import EnvParams
from flow.core.vehicles import Vehicles
from flow.controllers import RLController
from flow.core.rewards import average_velocity, total_velocity, min_delay, \
    desired_velocity, max_edge_velocity, penalize_standstill, \
    penalize_near_standstill, punish_small_rl_headways, \
    boolean_action_penalty

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
        """Test the max_edge_velocity method."""
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

    def test_min_delay(self):
        """Test the min_delay method."""
        # try the case of an environment with no vehicles
        vehicles = Vehicles()
        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # check that the reward function return 0 in the case of no vehicles
        self.assertEqual(min_delay(env), 0)

        # try the case of multiple vehicles
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)
        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # check the min_delay upon reset
        self.assertAlmostEqual(min_delay(env), 0)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 10)

        # check the min_delay with the new speed
        self.assertAlmostEqual(min_delay(env), 0.0333333333333)

    def test_penalize_standstill(self):
        """Test the penalize_standstill method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1})

        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params)

        # check the penalty is acknowledging all vehicles
        self.assertEqual(penalize_standstill(env, gain=1), -10)
        self.assertEqual(penalize_standstill(env, gain=2), -20)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 10)

        # check the penalty is acknowledging all vehicles but one
        self.assertEqual(penalize_standstill(env, gain=1), -9)
        self.assertEqual(penalize_standstill(env, gain=2), -18)

    def test_penalize_near_standstill(self):
        """Test the penalize_near_standstill method."""
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)

        env_params = EnvParams(additional_params={
            "target_velocity": 10, "max_accel": 1, "max_decel": 1})

        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params)

        # check the penalty is acknowledging all vehicles
        self.assertEqual(penalize_near_standstill(env, gain=1), -10)
        self.assertEqual(penalize_near_standstill(env, gain=2), -20)

        # change the speed of one vehicle
        env.vehicles.test_set_speed("test_0", 1)

        # check the penalty with good and bad thresholds
        self.assertEqual(penalize_near_standstill(env, thresh=2), -10)
        self.assertEqual(penalize_near_standstill(env, thresh=0.5), -9)

    def test_punish_small_rl_headways(self):
        """Test the punish_small_rl_headways method."""
        vehicles = Vehicles()
        vehicles.add("test", acceleration_controller=(RLController, {}),
                     num_vehicles=10)

        env, scenario = ring_road_exp_setup(vehicles=vehicles)

        # set the headways to 0
        for veh_id in env.vehicles.get_rl_ids():
            env.vehicles.set_headway(veh_id, 0)

        # test penalty when headways of all vehicles are currently 0
        self.assertEqual(punish_small_rl_headways(env, headway_threshold=1),
                         -10)
        self.assertEqual(punish_small_rl_headways(env, headway_threshold=2),
                         -10)
        self.assertEqual(punish_small_rl_headways(env,
                                                  headway_threshold=2,
                                                  penalty_gain=2),
                         -20)
        self.assertEqual(punish_small_rl_headways(env,
                                                  headway_threshold=2,
                                                  penalty_gain=2,
                                                  penalty_exponent=2),
                         -20)

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
