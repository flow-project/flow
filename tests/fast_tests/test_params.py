import unittest
from flow.core.params import EnvParams, SumoParams, SumoLaneChangeParams, \
    SumoCarFollowingParams, VehicleParams, NetParams
from flow.envs import Env
from flow.scenarios import LoopScenario
import os
import numpy as np
from gym.spaces import Box

os.environ["TEST_FLAG"] = "True"


class RLActionsEnv(Env):
    """A class used when testing clipping actions."""

    def _apply_rl_actions(self, rl_actions):
        pass

    def get_state(self):
        return [1]

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,))

    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(1,))

    def compute_reward(self, rl_actions, **kwargs):
        """Returns what rl_actions is perceived as."""
        return rl_actions[0]


class TestEnvParams(unittest.TestCase):
    """Tests flow.core.params.EnvParams"""

    def test_clip_actions(self):
        """Test that the actions are clipped in the reward function."""
        sim_params = SumoParams()
        vehicles = VehicleParams()
        additional_net_params = {
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)
        scenario = LoopScenario(
            name="RingRoadTest",
            vehicles=vehicles,
            net_params=net_params)

        # when set to False
        env_params = EnvParams(clip_actions=False)
        env = RLActionsEnv(
            env_params=env_params, sim_params=sim_params, scenario=scenario)
        env.reset()
        _, ret, _, _ = env.step(rl_actions=[5])
        self.assertEqual(np.mean(ret), 5)

        # when set to True
        env_params = EnvParams(clip_actions=True)
        env = RLActionsEnv(
            env_params=env_params, sim_params=sim_params, scenario=scenario)
        env.reset()

        _, ret, _, _ = env.step(rl_actions=[0.5])
        self.assertEqual(ret, 0.5)

        _, ret, _, _ = env.step(rl_actions=[5])
        self.assertEqual(ret, 1)

        _, ret, _, _ = env.step(rl_actions=[-5])
        self.assertEqual(ret, -1)


class TestSumoParams(unittest.TestCase):
    """Tests flow.core.params.SumoParams"""

    def test_params(self):
        """Tests that the various parameters lead to correct assignments in the
        attribute of the class."""
        # start a SumoParams with some attributes
        params = SumoParams(
             port=None,
             sim_step=0.125,
             emission_path=None,
             lateral_resolution=None,
             no_step_log=False,
             render=True,
             save_render=True,
             sight_radius=50,
             show_radius=True,
             pxpm=10,
             overtake_right=True,
             seed=204,
             restart_instance=True,
             print_warnings=False,
             teleport_time=-1)

        # ensure that the attributes match their correct values
        self.assertEqual(params.port, None)
        self.assertEqual(params.sim_step, 0.125)
        self.assertEqual(params.emission_path, None)
        self.assertEqual(params.lateral_resolution, None)
        self.assertEqual(params.no_step_log, False)
        self.assertEqual(params.render, True)
        self.assertEqual(params.save_render, True)
        self.assertEqual(params.sight_radius, 50)
        self.assertEqual(params.show_radius, True)
        self.assertEqual(params.pxpm, 10)
        self.assertEqual(params.overtake_right, True)
        self.assertEqual(params.seed, 204)
        self.assertEqual(params.restart_instance, True)
        self.assertEqual(params.print_warnings, False)
        self.assertEqual(params.teleport_time, -1)


class TestSumoCarFollowingParams(unittest.TestCase):
    """Tests flow.core.params.SumoCarFollowingParams"""

    def test_params(self):
        """Tests that the various parameters lead to correct assignments in the
        controller_params attribute of the class."""
        # start a SumoCarFollowingParams with some attributes
        cfm_params = SumoCarFollowingParams(
            accel=1.0,
            decel=1.5,
            sigma=0.5,
            tau=0.5,
            min_gap=1.0,
            max_speed=30,
            speed_factor=1.0,
            speed_dev=0.1,
            impatience=0.5,
            car_follow_model="IDM")

        # ensure that the attributes match their correct element in the
        # "controller_params" dict
        self.assertEqual(cfm_params.controller_params["accel"], 1)
        self.assertEqual(cfm_params.controller_params["decel"], 1.5)
        self.assertEqual(cfm_params.controller_params["sigma"], 0.5)
        self.assertEqual(cfm_params.controller_params["tau"], 0.5)
        self.assertEqual(cfm_params.controller_params["minGap"], 1)
        self.assertEqual(cfm_params.controller_params["maxSpeed"], 30)
        self.assertEqual(cfm_params.controller_params["speedFactor"], 1)
        self.assertEqual(cfm_params.controller_params["speedDev"], 0.1)
        self.assertEqual(cfm_params.controller_params["impatience"], 0.5)
        self.assertEqual(cfm_params.controller_params["carFollowModel"], "IDM")

    def test_deprecated(self):
        """Ensures that deprecated forms of the attribute still return proper
        values to the correct attributes"""
        # start a SumoCarFollowingParams with some attributes, using the
        # deprecated attributes
        cfm_params = SumoCarFollowingParams(
            accel=1.0,
            decel=1.5,
            sigma=0.5,
            tau=0.5,
            minGap=1.0,
            maxSpeed=30,
            speedFactor=1.0,
            speedDev=0.1,
            impatience=0.5,
            carFollowModel="IDM")

        # ensure that the attributes match their correct element in the
        # "controller_params" dict
        self.assertEqual(cfm_params.controller_params["accel"], 1)
        self.assertEqual(cfm_params.controller_params["decel"], 1.5)
        self.assertEqual(cfm_params.controller_params["sigma"], 0.5)
        self.assertEqual(cfm_params.controller_params["tau"], 0.5)
        self.assertEqual(cfm_params.controller_params["minGap"], 1)
        self.assertEqual(cfm_params.controller_params["maxSpeed"], 30)
        self.assertEqual(cfm_params.controller_params["speedFactor"], 1)
        self.assertEqual(cfm_params.controller_params["speedDev"], 0.1)
        self.assertEqual(cfm_params.controller_params["impatience"], 0.5)
        self.assertEqual(cfm_params.controller_params["carFollowModel"], "IDM")


class TestSumoLaneChangeParams(unittest.TestCase):
    """Tests flow.core.params.SumoLaneChangeParams"""

    def test_lc_params(self):
        """Test basic usage of the SumoLaneChangeParams object. Ensures that
        the controller_params attribute contains different elements depending
        on whether LC2103 or SL2015 is being used as the model."""
        # test for LC2013
        lc_params_1 = SumoLaneChangeParams(model="LC2013")
        attributes_1 = list(lc_params_1.controller_params.keys())
        # TODO: modify with all elements once the fix is added to sumo
        expected_attributes_1 = [
            "laneChangeModel", "lcStrategic", "lcCooperative", "lcSpeedGain",
            "lcKeepRight"
        ]
        self.assertCountEqual(attributes_1, expected_attributes_1)

        # test for SL2015
        lc_params_2 = SumoLaneChangeParams(model="SL2015")
        attributes_2 = list(lc_params_2.controller_params.keys())
        expected_attributes_2 = \
            ["laneChangeModel", "lcStrategic", "lcCooperative", "lcSpeedGain",
             "lcKeepRight", "lcLookaheadLeft", "lcSpeedGainRight", "lcSublane",
             "lcPushy", "lcPushyGap", "lcAssertive", "lcImpatience",
             "lcTimeToImpatience", "lcAccelLat"]
        self.assertCountEqual(attributes_2, expected_attributes_2)

    def test_wrong_model(self):
        """Tests that a wrongly specified model defaults the sumo lane change
        model to LC2013."""
        # input a wrong lane change model
        lc_params = SumoLaneChangeParams(model="foo")

        # ensure that the model is set to "LC2013"
        self.assertEqual(lc_params.controller_params["laneChangeModel"],
                         "LC2013")

        # ensure that the correct parameters are currently present
        attributes = list(lc_params.controller_params.keys())
        expected_attributes = [
            "laneChangeModel", "lcStrategic", "lcCooperative", "lcSpeedGain",
            "lcKeepRight"
        ]
        self.assertCountEqual(attributes, expected_attributes)

    def test_deprecated(self):
        """Ensures that deprecated forms of the attribute still return proper
        values to the correct attributes"""
        # start a SumoLaneChangeParams with some attributes
        lc_params = SumoLaneChangeParams(
            model="SL2015",
            lcStrategic=1.0,
            lcCooperative=1.0,
            lcSpeedGain=1.0,
            lcKeepRight=1.0,
            lcLookaheadLeft=2.0,
            lcSpeedGainRight=1.0,
            lcSublane=1.0,
            lcPushy=0,
            lcPushyGap=0.6,
            lcAssertive=1,
            lcImpatience=0,
            lcTimeToImpatience=float("inf"))

        # ensure that the attributes match their correct element in the
        # "controller_params" dict
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcStrategic"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcCooperative"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcSpeedGain"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcKeepRight"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcSublane"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcPushy"]), 0)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcPushyGap"]), 0.6)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcAssertive"]), 1)
        self.assertAlmostEqual(
            float(lc_params.controller_params["lcImpatience"]), 0)


if __name__ == '__main__':
    unittest.main()
