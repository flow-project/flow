import unittest
import os
from flow.core.vehicles import Vehicles
from flow.core.params import NetParams, EnvParams, SumoParams
from flow.scenarios import LoopScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_PARAMS
from flow.envs import TestEnv


os.environ["TEST_FLAG"] = "True"


class LaneChangeAccelEnv(unittest.TestCase):

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


class LaneChangeAccelPOEnv(unittest.TestCase):

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


class TestAccelEnv(unittest.TestCase):

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


class TestWaveAttenuationPOEnv(unittest.TestCase):

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


class TestWaveAttenuationMergeEnv(unittest.TestCase):

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
