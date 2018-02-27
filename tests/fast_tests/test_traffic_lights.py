import unittest
import os
os.environ["TEST_FLAG"] = "True"

from flow.core.params import NetParams
from flow.core.traffic_lights import TrafficLights
from tests.setup_scripts import ring_road_exp_setup


class TestUpdateGetState(unittest.TestCase):
    """
    Tests the update and get_state functions are working properly.
    """

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_single_lane(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with one lane
        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

        self.env.reset()
        self.env._step([])

        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "G")

    def test_multi_lane(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with two lanes
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

        self.env.reset()
        self.env.step([])

        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "GG")


class TestSetState(unittest.TestCase):
    """
    Tests the set_state function
    """

    def setUp(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with two lanes
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_all_lanes(self):
        # reset the environment
        self.env.reset()

        # set all states to something
        self.env.traffic_lights.set_state(node_id="top", env=self.env,
                                          state="rY")

        # run a new step
        self.env.step([])

        # check the new values
        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "rY")

    def test_single_lane(self):
        # reset the environment
        self.env.reset()

        # set all state of lane 1 to something
        self.env.traffic_lights.set_state(node_id="top", env=self.env,
                                          state="R", link_index=1)

        # run a new step
        self.env.step([])

        # check the new values
        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state[1], "R")


if __name__ == '__main__':
    unittest.main()
