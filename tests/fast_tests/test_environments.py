import numpy as np
import unittest
import os
from flow.core.vehicles import Vehicles
from flow.core.params import NetParams, EnvParams, SumoParams
from flow.controllers import IDMController, RLController
from flow.scenarios import LoopScenario, MergeScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_PARAMS
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS as MERGE_PARAMS
from flow.envs import LaneChangeAccelEnv, LaneChangeAccelPOEnv, AccelEnv, \
    WaveAttenuationEnv, WaveAttenuationPOEnv, WaveAttenuationMergePOEnv, \
    TestEnv


os.environ["TEST_FLAG"] = "True"


class TestLaneChangeAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = LoopScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=LOOP_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "lane_change_duration": 5
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 3,
                  "lane_change_duration": 5,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "lane_change_duration": 5,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "lane_change_duration": 5}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = LaneChangeAccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertEqual(env.observation_space.shape[0],
                         3 * env.vehicles.num_vehicles)
        self.assertTrue(all(env.observation_space.high == 1))
        self.assertTrue(all(env.observation_space.low == 0))

        # check the action space
        self.assertEqual(env.action_space.shape[0],
                         2 * env.vehicles.num_rl_vehicles)
        self.assertTrue(
            (env.action_space.high ==
             np.array([env.env_params.additional_params["max_accel"],
                       1])).all())
        self.assertTrue(
            (env.action_space.low ==
             np.array([-env.env_params.additional_params["max_decel"],
                       -1])).all())

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        env = LaneChangeAccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )
        env.additional_command()
        self.assertListEqual(env.vehicles.get_observed_ids(),
                             env.vehicles.get_human_ids())
        env.terminate()


class TestLaneChangeAccelPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = LoopScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=LOOP_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "lane_change_duration": 5
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 3,
                  "lane_change_duration": 5,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "lane_change_duration": 5,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "lane_change_duration": 5}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            LaneChangeAccelPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = LaneChangeAccelPOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertEqual(env.observation_space.shape[0], 5)
        self.assertTrue(all(env.observation_space.high == 1))
        self.assertTrue(all(env.observation_space.low == 0))

        # check the action space
        self.assertEqual(env.action_space.shape[0], 2)
        self.assertTrue(all(env.action_space.high == np.array([3, 1])))
        self.assertTrue(all(env.action_space.low == np.array([-3, -1])))

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        env = LaneChangeAccelPOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )
        env.step(None)
        env.additional_command()
        self.assertListEqual(env.vehicles.get_observed_ids(),
                             env.vehicles.get_leader(
                                 env.vehicles.get_rl_ids()))
        env.terminate()


class TestAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = LoopScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=LOOP_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 3,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            AccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            AccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            AccelEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = AccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # test the observation space
        self.assertEqual(env.observation_space.shape[0],
                         2 * env.vehicles.num_vehicles)
        self.assertTrue(all(env.observation_space.high == 1))
        self.assertTrue(all(env.observation_space.low == 0))

        # test the action space
        self.assertEqual(env.action_space.shape[0],
                         env.vehicles.num_rl_vehicles)
        self.assertEqual(env.action_space.high,
                         env.env_params.additional_params["max_accel"])
        self.assertEqual(env.action_space.low,
                         -env.env_params.additional_params["max_decel"])

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        env = AccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )
        env.additional_command()
        self.assertListEqual(env.vehicles.get_observed_ids(),
                             env.vehicles.get_human_ids())
        env.terminate()


class TestTwoLoopsMergeEnv(unittest.TestCase):

    def setUp(self):
        # TODO
        pass

    def tearDown(self):
        # TODO
        pass

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        # TODO
        pass

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # TODO
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        # TODO
        pass


class TestWaveAttenuationEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = LoopScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=LOOP_PARAMS.copy()),
        )
        params = {
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270]
        }
        self.env_params = EnvParams(additional_params=params)

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 1,
                  "max_decel": 1}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = WaveAttenuationEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # test the observation space
        self.assertEqual(env.observation_space.shape[0],
                         2 * env.vehicles.num_vehicles)
        self.assertTrue(all(env.observation_space.high == 1))
        self.assertTrue(all(env.observation_space.low == 0))

        # test the action space
        self.assertEqual(env.action_space.shape[0],
                         env.vehicles.num_rl_vehicles)
        self.assertEqual(env.action_space.high,
                         env.env_params.additional_params["max_accel"])
        self.assertEqual(env.action_space.low,
                         -env.env_params.additional_params["max_decel"])

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        env = WaveAttenuationEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )
        env.additional_command()
        self.assertListEqual(env.vehicles.get_observed_ids(),
                             env.vehicles.get_human_ids())
        env.terminate()


class TestWaveAttenuationPOEnv(unittest.TestCase):

    def setUp(self):
        self.sumo_params = SumoParams()
        self.scenario = LoopScenario(
            name="test_merge",
            vehicles=Vehicles(),
            net_params=NetParams(additional_params=LOOP_PARAMS.copy()),
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 1,
                  "max_decel": 1}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationPOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # TODO
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        # TODO
        pass


class TestWaveAttenuationMergeEnv(unittest.TestCase):

    def setUp(self):
        self.sumo_params = SumoParams()
        self.scenario = MergeScenario(
            name="test_merge",
            vehicles=Vehicles(),
            net_params=NetParams(additional_params=MERGE_PARAMS.copy()),
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        params = {"max_decel": 3,
                  "target_velocity": 25,
                  "num_rl": 5}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationMergePOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "target_velocity": 25,
                  "num_rl": 5}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationMergePOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "num_rl": 5}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationMergePOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 25}
        env_params = EnvParams(additional_params=params)
        self.assertRaises(
            KeyError,
            WaveAttenuationMergePOEnv,
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # TODO
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        # TODO
        pass


class TestTestEnv(unittest.TestCase):

    """Tests the TestEnv environment in flow/envs/test.py"""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("test")
        net_params = NetParams(additional_params=LOOP_PARAMS)
        env_params = EnvParams()
        sumo_params = SumoParams()
        scenario = LoopScenario("test_loop",
                                vehicles=vehicles,
                                net_params=net_params)
        self.env = TestEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_obs_space(self):
        self.assertEqual(self.env.observation_space.shape[0], 0)
        self.assertEqual(len(self.env.observation_space.high), 0)
        self.assertEqual(len(self.env.observation_space.low), 0)

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape[0], 0)
        self.assertEqual(len(self.env.action_space.high), 0)
        self.assertEqual(len(self.env.action_space.low), 0)

    def test_get_state(self):
        self.assertEqual(len(self.env.get_state()), 0)

    def test_compute_reward(self):
        # test the default
        self.assertEqual(self.env.compute_reward([]), 0)

        # test if the "reward_fn" parameter is defined
        def reward_fn(*_):
            return 1

        self.env.env_params.additional_params["reward_fn"] = reward_fn
        self.assertEqual(self.env.compute_reward([]), 1)


if __name__ == '__main__':
    unittest.main()
