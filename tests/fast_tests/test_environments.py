import unittest
import os
from flow.core.vehicles import Vehicles
from flow.core.params import NetParams, EnvParams, SumoParams
from flow.scenarios import LoopScenario, MergeScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_PARAMS
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS as MERGE_PARAMS
from flow.envs import LaneChangeAccelEnv, LaneChangeAccelPOEnv, AccelEnv, \
    WaveAttenuationEnv, WaveAttenuationPOEnv, WaveAttenuationMergePOEnv, \
    TestEnv


os.environ["TEST_FLAG"] = "True"


class TestLaneChangeAccelEnv(unittest.TestCase):

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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 10,
                  "lane_change_duration": 5}
        env_params = EnvParams(additional_params=params)
        env = LaneChangeAccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        pass


class TestLaneChangeAccelPOEnv(unittest.TestCase):

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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 10,
                  "lane_change_duration": 5}
        env_params = EnvParams(additional_params=params)
        env = LaneChangeAccelPOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        pass


class TestAccelEnv(unittest.TestCase):

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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 10}
        env_params = EnvParams(additional_params=params)
        env = AccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        pass


class TestTwoLoopsMergeEnv(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        pass

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        pass


class TestWaveAttenuationEnv(unittest.TestCase):

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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 1,
                  "max_decel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        env = WaveAttenuationEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        pass


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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 1,
                  "max_decel": 1,
                  "ring_length": [220, 270]}
        env_params = EnvParams(additional_params=params)
        env = WaveAttenuationPOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
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
        # check when any param is missing a KeyError is raised
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

        # check that when all params are passed no error is raised
        params = {"max_accel": 3,
                  "max_decel": 3,
                  "target_velocity": 25,
                  "num_rl": 5}
        env_params = EnvParams(additional_params=params)
        env = WaveAttenuationMergePOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=env_params
        )
        env.terminate()

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        pass

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
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
