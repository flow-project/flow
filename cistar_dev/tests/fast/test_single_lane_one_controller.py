import unittest
import logging

from cistar_dev.core.exp import SumoExperiment
from cistar_dev.envs.loop import LoopEnvironment
from cistar_dev.scenarios.loop.loop_scenario import LoopScenario
from cistar_dev.controllers.car_following_models import CFMController
from cistar_dev.controllers.lane_change_controllers import \
    never_change_lanes_controller


class TestSingleLaneOneController(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARNING)

        self.sumo_params = {"time_step": 0.01, "human_sm": 1}
        self.sumo_binary = "sumo"
        self.type_params = {"cfm_slow": (
            5, (CFMController, {'v_des': 8}), None, 0)}
        self.env_params = {"target_velocity": 25}
        self.net_params = {"length": 200, "lanes": 1, "speed_limit": 35,
                           "resolution": 40, "net_path": "tests/debug/net/"}
        self.cfg_params = {"start_time": 0, "end_time": 1000,
                           "cfg_path": "tests/debug/cfg/"}
        self.initial_config = {"shuffle": False}

    def test_it_runs(self):
        scenario = LoopScenario("test-single-lane-one-controller",
                                self.type_params, self.net_params,
                                self.cfg_params, self.initial_config)

        exp = SumoExperiment(LoopEnvironment, self.env_params,
                             self.sumo_binary, self.sumo_params, scenario)

        exp.run(1, 1000)  # params: num_runs, num_steps


if __name__ == '__main__':
    unittest.main()
