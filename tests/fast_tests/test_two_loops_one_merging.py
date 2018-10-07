import os
import unittest

from flow.controllers import RLController, IDMController, StaticLaneChanger
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_merges import TwoLoopsMergePOEnv, \
    ADDITIONAL_ENV_PARAMS
from flow.scenarios.loop_merge.gen import TwoLoopOneMergingGenerator
from flow.scenarios.loop_merge.scenario import TwoLoopsOneMergingScenario

os.environ["TEST_FLAG"] = "True"


def two_loops_one_merging_exp_setup(vehicles=None):
    sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        vehicles = Vehicles()
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(StaticLaneChanger, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=1)
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            lane_change_controller=(StaticLaneChanger, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=5)
        vehicles.add(
            veh_id="merge-idm",
            acceleration_controller=(IDMController, {}),
            lane_change_controller=(StaticLaneChanger, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=5)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = {
        "ring_radius": 50,
        "lane_length": 75,
        "inner_lanes": 3,
        "outer_lanes": 2,
        "speed_limit": 30,
        "resolution": 40
    }

    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="custom",
        lanes_distribution=1,
        additional_params={"merge_bunching": 0})

    scenario = TwoLoopsOneMergingScenario(
        "loop-merges",
        TwoLoopOneMergingGenerator,
        vehicles,
        net_params,
        initial_config=initial_config)

    env = TwoLoopsMergePOEnv(env_params, sumo_params, scenario)

    return env, scenario


class TestLoopMerges(unittest.TestCase):
    """
    Tests the loop_merges generator, scenario, and environment.
    """

    def setUp(self):
        # create the environment and scenario classes for a ring road
        self.env, scenario = two_loops_one_merging_exp_setup()

        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, scenario)

    def tearDown(self):
        # terminate the traci instance
        try:
            self.env.terminate()
        except FileNotFoundError:
            pass

        # free up used memory
        self.env = None
        self.exp = None

    def test_it_runs(self):
        """
        Tests that the loop merges experiment runs, and vehicles do not exit
        the network.
        """
        self.exp.run(1, 10)

    def test_gen_custom_start_pos(self):
        """
        Tests that vehicle with the prefix "merge" are in the merge_in lane,
        and all other vehicles are in the ring road.
        """
        # reset the environment to ensure all vehicles are at their starting
        # positions
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        # collect the starting edges of all vehicles
        merge_starting_edges = []
        other_starting_edges = []
        for veh_id in ids:
            if veh_id[:5] == "merge":
                merge_starting_edges.append(self.env.vehicles.get_edge(veh_id))
            else:
                other_starting_edges.append(self.env.vehicles.get_edge(veh_id))

        # ensure that all vehicles are starting in the edges they should be in
        expected_merge_starting_edges = ["right", "top", "bottom"]

        self.assertTrue(
            all(starting_edge in expected_merge_starting_edges
                for starting_edge in merge_starting_edges))

        self.assertTrue(
            all(starting_edge not in expected_merge_starting_edges
                for starting_edge in other_starting_edges))


if __name__ == '__main__':
    unittest.main()
