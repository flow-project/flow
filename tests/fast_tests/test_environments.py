import random
import numpy as np
import unittest
import os
from flow.core.params import VehicleParams
from flow.core.params import NetParams, EnvParams, SumoParams, InFlows
from flow.controllers import IDMController, RLController
from flow.scenarios import LoopScenario, MergeScenario, BottleneckScenario, \
    TwoLoopsOneMergingScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_PARAMS
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS as MERGE_PARAMS
from flow.scenarios.loop_merge import ADDITIONAL_NET_PARAMS as LM_PARAMS
from flow.envs import LaneChangeAccelEnv, LaneChangeAccelPOEnv, AccelEnv, \
    WaveAttenuationEnv, WaveAttenuationPOEnv, WaveAttenuationMergePOEnv, \
    TestEnv, TwoLoopsMergePOEnv, DesiredVelocityEnv


os.environ["TEST_FLAG"] = "True"


class TestLaneChangeAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestLaneChangeAccelPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelPOEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=AccelEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestTwoLoopsMergeEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=TwoLoopsMergePOEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )

    def test_reset(self):
        """
        Tests that the reset method creating new ring lengths within the
        requested range.
        """
        # set a random seed to ensure the network lengths are always the same
        # during testing
        random.seed(9001)

        # create the environment
        env = WaveAttenuationEnv(
            sim_params=self.sim_params,
            scenario=self.scenario,
            env_params=self.env_params
        )

        # reset the network several times and check its length
        self.assertEqual(env.scenario.length, 230)
        env.reset()
        self.assertEqual(env.scenario.length, 222)
        env.reset()
        self.assertEqual(env.scenario.length, 239)


class TestWaveAttenuationPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationPOEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestWaveAttenuationMergePOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        self.sim_params = SumoParams()
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
        self.sim_params = None
        self.scenario = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationMergePOEnv,
                sim_params=self.sim_params,
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
            sim_params=self.sim_params,
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
                sim_params=self.sim_params,
                scenario=self.scenario,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestTestEnv(unittest.TestCase):

    """Tests the TestEnv environment in flow/envs/test.py"""

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("test")
        net_params = NetParams(additional_params=LOOP_PARAMS)
        env_params = EnvParams()
        sim_params = SumoParams()
        scenario = LoopScenario("test_loop",
                                vehicles=vehicles,
                                net_params=net_params)
        self.env = TestEnv(env_params, sim_params, scenario)

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


class TestDesiredVelocityEnv(unittest.TestCase):

    """Tests the DesiredVelocityEnv environment in flow/envs/bottleneck.py"""

    def test_reset_inflows(self):
        """Tests that the inflow  change within the expected range when calling
        reset."""
        # set a random seed for inflows to be the same every time
        np.random.seed(seed=123)

        sim_params = SumoParams(sim_step=0.5, restart_instance=True)

        vehicles = VehicleParams()
        vehicles.add(veh_id="human")
        vehicles.add(veh_id="followerstopper")

        # edge name, how many segments to observe/control, whether the segment
        # is controlled
        controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                               ("4", 2, True), ("5", 1, False)]
        num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3),
                                 ("5", 1)]
        env_params = EnvParams(
            additional_params={
                "target_velocity": 40,
                "disable_tb": True,
                "disable_ramp_metering": True,
                "controlled_segments": controlled_segments,
                "symmetric": False,
                "observed_segments": num_observed_segments,
                "reset_inflow": True,  # this must be set to True for the test
                "lane_change_duration": 5,
                "max_accel": 3,
                "max_decel": 3,
                "inflow_range": [1000, 2000]  # this is what we're testing
            }
        )

        inflow = InFlows()
        inflow.add(veh_type="human",
                   edge="1",
                   vehs_per_hour=1500,  # the initial inflow we're checking for
                   departLane="random",
                   departSpeed=10)

        net_params = NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params={"scaling": 1})

        scenario = BottleneckScenario(
            name="bay_bridge_toll",
            vehicles=vehicles,
            net_params=net_params)

        env = DesiredVelocityEnv(env_params, sim_params, scenario)

        # reset the environment and get a new inflow rate
        env.reset()
        expected_inflow = 1353.6  # just from checking the new inflow

        # check that the first inflow rate is approximately 1500
        for _ in range(500):
            env.step(rl_actions=None)
        self.assertAlmostEqual(
            env.vehicles.get_inflow_rate(250)/expected_inflow, 1, 1)

        # reset the environment and get a new inflow rate
        env.reset()
        expected_inflow = 1756.8  # just from checking the new inflow

        # check that the new inflow rate is approximately as expected
        for _ in range(500):
            env.step(rl_actions=None)
        self.assertAlmostEqual(
            env.vehicles.get_inflow_rate(250)/expected_inflow, 1, 1)


###############################################################################
#                              Utility methods                                #
###############################################################################

def test_additional_params(env_class,
                           sim_params,
                           scenario,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    scenario : flow.scenarios.Scenario
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
                sim_params=sim_params,
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
                  sim_params,
                  scenario,
                  env_params,
                  expected_observed):
    """Test that the observed vehicles in the environment are as expected.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    scenario : flow.scenarios.Scenario
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
    env = env_class(sim_params=sim_params,
                    scenario=scenario,
                    env_params=env_params)
    env.reset()
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
