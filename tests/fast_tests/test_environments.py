import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, \
    NetParams, SumoCarFollowingParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS

from tests.setup_scripts import ring_road_exp_setup
import os
import numpy as np

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
