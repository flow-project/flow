import unittest
import logging

from cistar.core.exp import SumoExperiment
from cistar.envs.loop_accel import SimpleAccelerationEnvironment
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.controllers.car_following_models import CFMController
from cistar.controllers.lane_change_controllers import *


class TestTwoLaneTwoController(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        self.sumo_params = {"time_step": 0.01, "traci_control": 1}
        self.sumo_binary = "sumo"
        self.type_params = { "cfm-slow": (6, (CFMController, {'v_des': 6}), (StaticLaneChanger, {}), 0),\
                            "cfm-fast": (6, (CFMController, {'v_des': 10}), (StochasticLaneChanger, {}), 0)}
        self.env_params = {"target_velocity": 8}
        self.net_params = {"length": 200, "lanes": 2, "speed_limit": 35,
                           "resolution": 40, "net_path": "tests/debug/net/"}
        self.cfg_params = {"start_time": 0, "end_time": 1000,
                           "cfg_path": "tests/debug/cfg/"}

    def test_it_runs(self):
        scenario = LoopScenario("test-two-lane-two-controller",
                                self.type_params, self.net_params,
                                self.cfg_params)

        # FIXME(cathywu) it currently looks like there's no lane changing,
        # although there should be.
        # self.sumo_binary = "sumo-gui"
        exp = SumoExperiment(SimpleAccelerationEnvironment, self.env_params,
                             self.sumo_binary, self.sumo_params, scenario)

        exp.run(1, 1000)  # params: num_runs, num_steps

        exp.env.terminate()  # Dummy test


if __name__ == '__main__':
    unittest.main()
