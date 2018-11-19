import numpy as np
import unittest
import os
from flow.core.vehicles import Vehicles
from flow.core.params import NetParams, EnvParams, SumoParams
from flow.controllers import IDMController, RLController
from flow.scenarios import LoopScenario, MergeScenario, \
    TwoLoopsOneMergingScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_PARAMS
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS as MERGE_PARAMS
from flow.scenarios.loop_merge import ADDITIONAL_NET_PARAMS as LM_PARAMS
from flow.envs import LaneChangeAccelEnv, LaneChangeAccelPOEnv, AccelEnv, \
    WaveAttenuationEnv, WaveAttenuationPOEnv, WaveAttenuationMergePOEnv, \
    TestEnv, TwoLoopsMergePOEnv


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
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "lane_change_duration": 5,
                    "target_velocity": 10
                }
            )
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
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=3 * env.vehicles.num_vehicles,
            expected_min=0,
            expected_max=1)
        )

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=2 * env.vehicles.num_rl_vehicles,
            expected_min=np.array([
                -env.env_params.additional_params["max_decel"], -1]),
            expected_max=np.array([
                env.env_params.additional_params["max_accel"], 1]))
        )

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=LaneChangeAccelEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


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
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelPOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "lane_change_duration": 5,
                    "target_velocity": 10
                }
            )
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
        self.assertTrue(test_space(
            env.observation_space, expected_size=5, expected_min=0,
            expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=2,
            expected_min=np.array([-3, -1]),
            expected_max=np.array([3, 1]))
        )

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=LaneChangeAccelPOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


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
        self.assertTrue(
            test_additional_params(
                env_class=AccelEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 10
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = AccelEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=2 * env.vehicles.num_vehicles,
            expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=env.vehicles.num_rl_vehicles,
            expected_min=-abs(env.env_params.additional_params["max_decel"]),
            expected_max=env.env_params.additional_params["max_accel"])
        )

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=AccelEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestTwoLoopsMergeEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = TwoLoopsOneMergingScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(
                no_internal_links=False,
                additional_params=LM_PARAMS.copy(),
            ),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "n_preceding": 2,
                "n_following": 2,
                "n_merging_in": 2,
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=TwoLoopsMergePOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 3,
                    "target_velocity": 10,
                    "n_preceding": 2,
                    "n_following": 2,
                    "n_merging_in": 2
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = TwoLoopsMergePOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=17, expected_min=0, expected_max=float('inf')))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=env.vehicles.num_rl_vehicles,
            expected_min=-abs(env.env_params.additional_params["max_decel"]),
            expected_max=env.env_params.additional_params["max_accel"])
        )

        env.terminate()


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
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "ring_length": [220, 270],
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = WaveAttenuationEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=2 * env.vehicles.num_vehicles,
            expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=env.vehicles.num_rl_vehicles,
            expected_min=-abs(env.env_params.additional_params["max_decel"]),
            expected_max=env.env_params.additional_params["max_accel"])
        )

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=WaveAttenuationEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestWaveAttenuationPOEnv(unittest.TestCase):

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
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270]
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationPOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "ring_length": [220, 270],
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = WaveAttenuationPOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=3, expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=1, expected_min=-1, expected_max=1))

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=WaveAttenuationPOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestWaveAttenuationMergePOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sumo_params = SumoParams()
        self.scenario = MergeScenario(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=MERGE_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 25,
                "num_rl": 5,
            }
        )

    def tearDown(self):
        self.sumo_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationMergePOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 25,
                    "num_rl": 5
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = WaveAttenuationMergePOEnv(
            sumo_params=self.sumo_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=25, expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=5, expected_min=-3, expected_max=3))

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=WaveAttenuationMergePOEnv,
                sumo_params=self.sumo_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


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


###############################################################################
#                              Utility methods                                #
###############################################################################

def test_additional_params(env_class,
                           sumo_params,
                           scenario,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sumo_params : flow.scenarios.Scenario
        sumo-specific parameters
    scenario : flow.core.params.SumoParams
        scenario that works for the environment
    additional_params : dict
        the valid and required additional parameters for the environment in
        EnvParams

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    for key in additional_params.keys():
        # remove one param from the additional_params dict
        new_add = additional_params.copy()
        del new_add[key]

        try:
            env_class(
                sumo_params=sumo_params,
                scenario=scenario,
                env_params=EnvParams(additional_params=new_add)
            )
            # if no KeyError is raised, the test has failed, so return False
            return False
        except KeyError:
            # if a KeyError is raised, test the next param
            pass

    # if removing all additional params led to KeyErrors, the test has passed,
    # so return True
    return True


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test that an action or observation space is the correct size and bounds.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or numpy.ndarray
        expected minimum value(s)
    expected_max : float or numpy.ndarray
        expected maximum value(s)

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    return gym_space.shape[0] == expected_size \
        and all(gym_space.high == expected_max) \
        and all(gym_space.low == expected_min)


def test_observed(env_class,
                  sumo_params,
                  scenario,
                  env_params,
                  expected_observed):
    """Test that the observed vehicles in the environment are as expected.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sumo_params : flow.scenarios.Scenario
        sumo-specific parameters
    scenario : flow.core.params.SumoParams
        scenario that works for the environment
    env_params : flow.core.params.EnvParams
        environment-specific parameters
    expected_observed : list or numpy.ndarray
        expected list of observed vehicles

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    env = env_class(sumo_params=sumo_params,
                    scenario=scenario,
                    env_params=env_params)
    env.step(None)
    env.additional_command()
    test_mask = np.all(
        np.array(env.vehicles.get_observed_ids()) ==
        np.array(expected_observed)
    )
    env.terminate()

    return test_mask

###############################################################################
#                                End of utils                                 #
###############################################################################


if __name__ == '__main__':
    unittest.main()
