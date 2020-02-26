import random
import numpy as np
import unittest
import os
from scipy.optimize import fsolve
from copy import deepcopy
from flow.core.params import VehicleParams
from flow.core.params import NetParams, EnvParams, SumoParams, InFlows
from flow.controllers import IDMController, RLController
from flow.networks import RingNetwork, MergeNetwork, BottleneckNetwork
from flow.networks import HighwayRampsNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS as RING_PARAMS
from flow.networks.merge import ADDITIONAL_NET_PARAMS as MERGE_PARAMS
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS as \
    HIGHWAY_PARAMS
from flow.envs import LaneChangeAccelEnv, LaneChangeAccelPOEnv, AccelEnv, \
    WaveAttenuationEnv, WaveAttenuationPOEnv, MergePOEnv, \
    TestEnv, BottleneckDesiredVelocityEnv, BottleneckEnv, BottleneckAccelEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function
from flow.envs.multiagent import MultiAgentHighwayPOEnv
from flow.envs.multiagent import MultiAgentAccelPOEnv
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.envs.multiagent import MultiAgentMergePOEnv

os.environ["TEST_FLAG"] = "True"


class TestLaneChangeAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "lane_change_duration": 5,
                "sort_vehicles": False
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "lane_change_duration": 5,
                    "target_velocity": 10,
                    "sort_vehicles": False
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = LaneChangeAccelEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=3 * env.initial_vehicles.num_vehicles,
            expected_min=0,
            expected_max=1)
        )

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=2 * env.initial_vehicles.num_rl_vehicles,
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
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestLaneChangeAccelPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "lane_change_duration": 5,
                "sort_vehicles": False
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=LaneChangeAccelPOEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "lane_change_duration": 5,
                    "target_velocity": 10,
                    "sort_vehicles": False
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = LaneChangeAccelPOEnv(
            sim_params=self.sim_params,
            network=self.network,
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
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestAccelEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "target_velocity": 10,
                "sort_vehicles": False
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=AccelEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 10,
                    "sort_vehicles": False
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        env = AccelEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=2 * env.initial_vehicles.num_vehicles,
            expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=env.initial_vehicles.num_rl_vehicles,
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
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )

    def test_sorting(self):
        """
        Tests that the sorting method returns a list of ids sorted by the
        absolute_position variable when sorting is requested, and does
        nothing if it is not requested.
        """
        env_params = self.env_params
        env_params.additional_params['sort_vehicles'] = True
        self.network.initial_config.shuffle = True

        env = AccelEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=env_params
        )

        env.reset()
        env.additional_command()

        sorted_ids = env.sorted_ids
        positions = [env.absolute_position[veh_id] for veh_id in sorted_ids]

        # ensure vehicles ids are in sorted order by positions
        self.assertTrue(
            all(positions[i] <= positions[i + 1]
                for i in range(len(positions) - 1)))

    def test_no_sorting(self):
        # setup a environment with the "sort_vehicles" attribute set to False,
        # and shuffling so that the vehicles are not sorted by their ids
        env_params = self.env_params
        env_params.additional_params['sort_vehicles'] = False
        self.network.initial_config.shuffle = True

        env = AccelEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=env_params
        )

        env.reset()
        env.additional_command()

        sorted_ids = list(env.sorted_ids)
        ids = env.k.vehicle.get_ids()

        # ensure that the list of ids did not change
        self.assertListEqual(sorted_ids, ids)


class TestWaveAttenuationEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams(
            restart_instance=True
        )
        self.network = RingNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        params = {
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270]
        }
        self.env_params = EnvParams(additional_params=params)

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationEnv,
                sim_params=self.sim_params,
                network=self.network,
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
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=2 * env.initial_vehicles.num_vehicles,
            expected_min=0, expected_max=1))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=env.initial_vehicles.num_rl_vehicles,
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
                network=self.network,
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
            network=self.network,
            env_params=self.env_params
        )

        # reset the network several times and check its length
        self.assertEqual(env.k.network.non_internal_length(), 230)
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), 239)
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), 256)

    def test_v_eq_max_function(self):
        """
        Tests that the v_eq_max_function returns appropriate values.
        """
        # for 230 m ring roads
        self.assertAlmostEqual(
            float(fsolve(v_eq_max_function, np.array([4]), args=(22, 230))[0]),
            3.7136148111012934)

        # for 270 m ring roads
        self.assertAlmostEqual(
            float(fsolve(v_eq_max_function, np.array([4]), args=(22, 270))[0]),
            5.6143732387852054)

    def test_reset_no_same_length(self):
        """
        Tests that the reset method uses the original ring length when the
        range is set to None.
        """
        # setup env_params with not range
        env_params = deepcopy(self.env_params)
        env_params.additional_params["ring_length"] = None

        # create the environment
        env = WaveAttenuationEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=env_params
        )

        # reset the network several times and check its length
        self.assertEqual(env.k.network.non_internal_length(), RING_PARAMS["length"])
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), RING_PARAMS["length"])
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), RING_PARAMS["length"])


class TestWaveAttenuationPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
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
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=WaveAttenuationPOEnv,
                sim_params=self.sim_params,
                network=self.network,
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
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=3,
            expected_min=-float('inf'),
            expected_max=float('inf')
        ))

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
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )

    def test_reward(self):
        """Check the reward function for different values.

        The reward function should be a linear combination of the average speed
        of all vehicles and a penalty on the requested accelerations by the
        AVs.
        """
        # create the environment
        env = WaveAttenuationPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )
        env.reset()

        # check the reward for no acceleration

        env.k.vehicle.test_set_speed('human_0', 0)
        env.k.vehicle.test_set_speed('rl_0', 0)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=[0], fail=False),
            0
        )

        env.k.vehicle.test_set_speed('human_0', 0)
        env.k.vehicle.test_set_speed('rl_0', 1)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=[0], fail=False),
            0.1
        )

        env.k.vehicle.test_set_speed('human_0', 1)
        env.k.vehicle.test_set_speed('rl_0', 1)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=[0], fail=False),
            0.2
        )

        # check the fail option

        env.k.vehicle.test_set_speed('human_0', 1)
        env.k.vehicle.test_set_speed('rl_0', 1)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=[0], fail=True),
            0
        )

        # check the effect of RL actions

        env.k.vehicle.test_set_speed('human_0', 1)
        env.k.vehicle.test_set_speed('rl_0', 1)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=None, fail=False),
            0
        )

        env.k.vehicle.test_set_speed('human_0', 1)
        env.k.vehicle.test_set_speed('rl_0', 1)
        self.assertAlmostEqual(
            env.compute_reward(rl_actions=[1], fail=False),
            -3.8
        )


class TestMergePOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = MergeNetwork(
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
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=MergePOEnv,
                sim_params=self.sim_params,
                network=self.network,
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
        env = MergePOEnv(
            sim_params=self.sim_params,
            network=self.network,
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
                env_class=MergePOEnv,
                sim_params=self.sim_params,
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestTestEnv(unittest.TestCase):

    """Tests the TestEnv environment in flow/envs/test.py"""

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("test", num_vehicles=1)
        net_params = NetParams(additional_params=RING_PARAMS)
        env_params = EnvParams()
        sim_params = SumoParams()
        network = RingNetwork("test_ring",
                              vehicles=vehicles,
                              net_params=net_params)
        self.env = TestEnv(env_params, sim_params, network)

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


class TestBottleneckEnv(unittest.TestCase):

    """Tests the BottleneckEnv environment in flow/envs/bottleneck.py"""

    def setUp(self):
        self.sim_params = SumoParams(sim_step=0.5, restart_instance=True)

        vehicles = VehicleParams()
        vehicles.add(veh_id="human", num_vehicles=10)

        env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "lane_change_duration": 5,
                "disable_tb": True,
                "disable_ramp_metering": True,
            }
        )

        net_params = NetParams(
            additional_params={"scaling": 1, "speed_limit": 23})

        self.network = BottleneckNetwork(
            name="bay_bridge_toll",
            vehicles=vehicles,
            net_params=net_params)

        self.env = BottleneckEnv(env_params, self.sim_params, self.network)
        self.env.reset()

    def tearDown(self):
        self.env.terminate()
        del self.env

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=BottleneckEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "lane_change_duration": 5,
                    "disable_tb": True,
                    "disable_ramp_metering": True,
                }
            )
        )

    def test_get_bottleneck_density(self):
        self.assertEqual(self.env.get_bottleneck_density(), 0)

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # check the observation space
        self.assertTrue(test_space(
            self.env.observation_space,
            expected_size=1,
            expected_min=-float('inf'),
            expected_max=float('inf'))
        )

        # check the action space
        self.assertTrue(test_space(
            self.env.action_space,
            expected_size=1,
            expected_min=-float('inf'),
            expected_max=float('inf'))
        )


class TestBottleneckAccelEnv(unittest.TestCase):

    """Tests BottleneckAccelEnv in flow/envs/bottleneck.py."""

    def setUp(self):
        self.sim_params = SumoParams(sim_step=0.5, restart_instance=True)

        vehicles = VehicleParams()
        vehicles.add(veh_id="human", num_vehicles=10)

        env_params = EnvParams(
            additional_params={
                "max_accel": 3,
                "max_decel": 3,
                "lane_change_duration": 5,
                "disable_tb": True,
                "disable_ramp_metering": True,
                "target_velocity": 30,
                "add_rl_if_exit": True,
            }
        )

        net_params = NetParams(
            additional_params={"scaling": 1, "speed_limit": 23})

        self.network = BottleneckNetwork(
            name="bay_bridge_toll",
            vehicles=vehicles,
            net_params=net_params)

        self.env = BottleneckAccelEnv(
            env_params, self.sim_params, self.network)
        self.env.reset()

    def tearDown(self):
        self.env.terminate()
        del self.env

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=BottleneckAccelEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 3,
                    "max_decel": 3,
                    "lane_change_duration": 5,
                    "disable_tb": True,
                    "disable_ramp_metering": True,
                    "target_velocity": 30,
                    "add_rl_if_exit": True,
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # check the observation space
        self.assertTrue(test_space(
            self.env.observation_space,
            expected_size=12,
            expected_min=0,
            expected_max=1)
        )


class TestBottleneckDesiredVelocityEnv(unittest.TestCase):

    """Tests the BottleneckDesiredVelocityEnv environment in
       flow/envs/bottleneck.py"""

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
            additional_params={"scaling": 1, "speed_limit": 23})

        network = BottleneckNetwork(
            name="bay_bridge_toll",
            vehicles=vehicles,
            net_params=net_params)

        env = BottleneckDesiredVelocityEnv(env_params, sim_params, network)

        # reset the environment and get a new inflow rate
        env.reset()
        expected_inflow = 1353.6  # just from checking the new inflow

        # check that the first inflow rate is approximately what the seeded
        # value expects it to be
        for _ in range(500):
            env.step(rl_actions=None)
        self.assertAlmostEqual(
            env.k.vehicle.get_inflow_rate(250)/expected_inflow, 1, 1)


class TestMultiAgentAccelPOEnv(unittest.TestCase):
    """Tests the MultiAgentAccelPOEnv environment in
       flow/envs/multiagent/ring/accel.py"""

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_ring",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                'max_accel': 1,
                'max_decel': 1,
                "target_velocity": 25
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=MultiAgentAccelPOEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 10,
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = MultiAgentAccelPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=6,
            expected_min=-5,
            expected_max=5
        ))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=1, expected_min=-1, expected_max=1))

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=MultiAgentMergePOEnv,
                sim_params=self.sim_params,
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestMultiAgentWaveAttenuationPOEnv(unittest.TestCase):
    """Tests the MultiAgentWaveAttenuationPOEnv environment in
       flow/envs/multiagent/ring/wave_attenuation.py"""

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = RingNetwork(
            name="test_ring",
            vehicles=vehicles,
            net_params=NetParams(additional_params=RING_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                'max_accel': 1,
                'max_decel': 1,
                "ring_length": [220, 270]
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=MultiAgentWaveAttenuationPOEnv,
                sim_params=self.sim_params,
                network=self.network,
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
        env = MultiAgentWaveAttenuationPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=3,
            expected_min=-5,
            expected_max=5
        ))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=1,
            expected_min=-1,
            expected_max=1
        ))

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=MultiAgentWaveAttenuationPOEnv,
                sim_params=self.sim_params,
                network=self.network,
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
        env = MultiAgentWaveAttenuationPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # reset the network several times and check its length
        self.assertEqual(env.k.network.non_internal_length(), 230)
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), 239)
        env.reset()
        self.assertEqual(env.k.network.non_internal_length(), 256)


class TestMultiAgentMergePOEnv(unittest.TestCase):
    """Tests the MultiAgentMergePOEnv environment in
       flow/envs/multiagent/merge.py"""

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = MergeNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=MERGE_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                'max_accel': 1,
                'max_decel': 1,
                "target_velocity": 25
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=MultiAgentMergePOEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 10,
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = MultiAgentMergePOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=5,
            expected_min=-5,
            expected_max=5
        ))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=1, expected_min=-1, expected_max=1))

        env.terminate()

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=MultiAgentMergePOEnv,
                sim_params=self.sim_params,
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


class TestMultiAgentHighwayPOEnv(unittest.TestCase):

    def setUp(self):
        vehicles = VehicleParams()
        vehicles.add("rl", acceleration_controller=(RLController, {}),
                     num_vehicles=1)
        vehicles.add("human", acceleration_controller=(IDMController, {}),
                     num_vehicles=1)

        self.sim_params = SumoParams()
        self.network = HighwayRampsNetwork(
            name="test_merge",
            vehicles=vehicles,
            net_params=NetParams(additional_params=HIGHWAY_PARAMS.copy()),
        )
        self.env_params = EnvParams(
            additional_params={
                'max_accel': 1,
                'max_decel': 1,
                "target_velocity": 25
            }
        )

    def tearDown(self):
        self.sim_params = None
        self.network = None
        self.env_params = None

    def test_additional_env_params(self):
        """Ensures that not returning the correct params leads to an error."""
        self.assertTrue(
            test_additional_params(
                env_class=MultiAgentHighwayPOEnv,
                sim_params=self.sim_params,
                network=self.network,
                additional_params={
                    "max_accel": 1,
                    "max_decel": 1,
                    "target_velocity": 10,
                }
            )
        )

    def test_observation_action_space(self):
        """Tests the observation and action spaces upon initialization."""
        # create the environment
        env = MultiAgentHighwayPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )

        # check the observation space
        self.assertTrue(test_space(
            env.observation_space,
            expected_size=5,
            expected_min=-float('inf'),
            expected_max=float('inf')
        ))

        # check the action space
        self.assertTrue(test_space(
            env.action_space,
            expected_size=1, expected_min=-1, expected_max=1))

        env.terminate()

    def test_compute_reward(self):
        # create the environment
        env = MultiAgentHighwayPOEnv(
            sim_params=self.sim_params,
            network=self.network,
            env_params=self.env_params
        )
        env.reset()

        # test the no actions case
        self.assertDictEqual(env.compute_reward(None), {})

        # test the failure case
        self.assertDictEqual(env.compute_reward({"rl_0": 0}, fail=True),
                             {"rl_0": 0})

        # test the generic case
        env.k.vehicle.test_set_speed("rl_0", 5)
        self.assertDictEqual(env.compute_reward({"rl_0": 0}, fail=False),
                             {"rl_0": 0.09446148586775807})

        # test the evaluate case
        env.k.vehicle.test_set_speed("rl_0", 5)
        env.env_params.evaluate = True
        self.assertDictEqual(env.compute_reward({"rl_0": 0}, fail=False),
                             {"rl_0": 5})

    def test_observed(self):
        """Ensures that the observed ids are returning the correct vehicles."""
        self.assertTrue(
            test_observed(
                env_class=MultiAgentHighwayPOEnv,
                sim_params=self.sim_params,
                network=self.network,
                env_params=self.env_params,
                expected_observed=["human_0"]
            )
        )


###############################################################################
#                              Utility methods                                #
###############################################################################

def test_additional_params(env_class,
                           sim_params,
                           network,
                           additional_params):
    """Test that the environment raises an Error in any param is missing.

    Parameters
    ----------
    env_class : flow.envs.Env type
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
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
                network=network,
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
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
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
                  network,
                  env_params,
                  expected_observed):
    """Test that the observed vehicles in the environment are as expected.

    Parameters
    ----------
    env_class : flow.envs.Env class
        blank
    sim_params : flow.core.params.SumoParams
        sumo-specific parameters
    network : flow.networks.Network
        network that works for the environment
    env_params : flow.core.params.EnvParams
        environment-specific parameters
    expected_observed : array_like
        expected list of observed vehicles

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    env = env_class(sim_params=sim_params,
                    network=network,
                    env_params=env_params)
    env.reset()
    env.step(None)
    env.additional_command()
    test_mask = np.all(
        np.array(env.k.vehicle.get_observed_ids()) ==
        np.array(expected_observed)
    )
    env.terminate()

    return test_mask

###############################################################################
#                                End of utils                                 #
###############################################################################


if __name__ == '__main__':
    unittest.main()
