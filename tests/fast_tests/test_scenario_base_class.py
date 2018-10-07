import unittest
import os
import numpy as np

from flow.core.params import InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController

from tests.setup_scripts import ring_road_exp_setup, figure_eight_exp_setup, \
    highway_exp_setup
from tests.setup_scripts import variable_lanes_exp_setup

os.environ["TEST_FLAG"] = "True"


class TestGetX(unittest.TestCase):
    """
    Tests the get_x function for vehicles placed in links and in junctions.
    This is tested on a scenario whose edgestarts are known beforehand
    (figure 8).
    """

    def setUp(self):
        # create the environment and scenario classes for a figure eight
        env, self.scenario = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def test_getx(self):
        # test for an edge in the lanes
        edge_1 = "bottom_lower_ring"
        pos_1 = 4.72
        self.assertAlmostEqual(self.scenario.get_x(edge_1, pos_1), 5)

        # test for an edge in the internal links
        edge_2 = ":bottom_lower_ring"
        pos_2 = 0.1
        self.assertAlmostEqual(self.scenario.get_x(edge_2, pos_2), 0.1)


class TestGetEdge(unittest.TestCase):
    """
    Tests the get_edge function for vehicles placed in links and in internal
    edges. This is tested on a scenario whose edgestarts are known beforehand
    (figure 8).
    """

    def setUp(self):
        # create the environment and scenario classes for a figure eight
        env, self.scenario = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def test_get_edge(self):
        # test for a position in the lanes
        x1 = 5
        self.assertTupleEqual(
            self.scenario.get_edge(x1), ("bottom_lower_ring", 4.72))

        # test for a position in the internal links
        x2 = 0.1
        self.assertTupleEqual(
            self.scenario.get_edge(x2), (":bottom_lower_ring", 0.1))


class TestEvenStartPos(unittest.TestCase):
    """
    Tests the function gen_even_start_pos in base_scenario.py. This function
    can be used on any scenario subclass, and therefore may be tested on any of
    these classes. In order to perform this testing, replace the scenario in
    setUp() with the scenario to be tested.
    """

    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        """
        Replace with any scenario you would like to test gen_even_start_pos on.
        In ordering for all the tests to be meaningful, the scenario must
        contain MORE THAN TWO LANES.
        """
        # create a multi-lane ring road network
        additional_net_params = {
            "length": 230,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=15)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(
            net_params=net_params,
            initial_config=initial_config,
            vehicles=vehicles)

    def tearDown_gen_start_pos(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_base(self):
        """
        Tests that get_even_start_pos function evenly distributed vehicles in a
        network.
        """
        # set the initial_config parameters as default (even spacing, no extra
        # conditions), and reset
        initial_config = InitialConfig(lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.scenario.length)

        # check that the position of the first vehicle is at 0
        self.assertEqual(veh_pos[0], 0)

        # if all element are equal, there should only be one unique value
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_x0(self):
        """
        Tests that the vehicles are uniformly distributed and the initial
        vehicle if at a given x0.
        """
        # set the initial_config parameters with an x0 value that is something
        # in between zero and the length of the network
        x0 = 10
        initial_config = InitialConfig(x0=x0, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.scenario.length)

        # check that the position of the first vehicle is at 0
        self.assertEqual(veh_pos[0], x0 % self.env.scenario.length)

        # if all element are equal, there should only be one unique value
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_bunching(self):
        """
        Tests that vehicles are uniformly distributed given a certain bunching
        """
        # set the initial_config parameters with a modest bunching term
        bunching = 10
        initial_config = InitialConfig(bunching=bunching, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.scenario.length)

        # check that all vehicles except the last vehicle have the same spacing
        self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # check that the spacing of the last vehicle is just offset by the
        # bunching term
        self.assertAlmostEqual(nth_headway[-1] - nth_headway[-2], bunching)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_bunching_too_small(self):
        """
        Tests that if bunching is negative, it is set to 0
        """
        # set the initial_config parameters with a negative bunching term
        bunching = -10
        initial_config = InitialConfig(bunching=bunching, lanes_distribution=1)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.scenario.length)

        # check that all vehicles, including the last vehicle, have the same
        # spacing
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution(self):
        """
        Tests that if lanes_distribution is less than the total number of
        lanes, the vehicles are uniformly distributed over the specified number
        of lanes.
        """
        # set the initial_config parameters with a lanes distribution less than
        # the number of lanes in the network.
        lanes_distribution = 2
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = []
        for i in range(self.env.scenario.lanes):
            veh_pos.append([
                self.env.get_x_by_id(veh_id) for veh_id in ids
                if self.env.vehicles.get_lane(veh_id) == i
            ])

        # check that the vehicles are uniformly distributed in the number of
        # requested lanes
        for i in range(lanes_distribution):
            # difference in position between the nth vehicle and the vehicle
            # ahead of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.scenario.length)

            self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # check that there are no vehicles in the remaining lanes
        for i in range(self.env.scenario.lanes - lanes_distribution):
            self.assertEqual(len(veh_pos[i + lanes_distribution]), 0)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution_too_small(self):
        """
        Tests that when lanes_distribution is less than 1, the number is set
        to 1.
        """
        # set the initial_config parameters with a small or negative
        # lanes_distribution
        lanes_distribution = np.random.randint(-100, 0)
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that only the first lane has vehicles
        ids = self.env.vehicles.get_ids()
        veh_lanes = [self.env.vehicles.get_lane(veh_id) for veh_id in ids]
        self.assertEqual(np.unique(veh_lanes).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution_too_large(self):
        """
        Tests that when lanes_distribution is greater than the number of lanes,
        the vehicles are distributed over the maximum number of lanes instead.
        """
        # set the initial_config parameter with a very large lanes_distribution
        lanes_distribution = np.inf
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = []
        for i in range(self.env.scenario.lanes):
            veh_pos.append([
                self.env.get_x_by_id(veh_id) for veh_id in ids
                if self.env.vehicles.get_lane(veh_id) == i
            ])

        # check that the vehicles are uniformly distributed in the number of
        # requested lanes lanes
        for i in range(self.env.scenario.lanes):
            # difference in position between the nth vehicle and the vehicle
            # ahead of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.scenario.length)

            self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_edges_distribution(self):
        """
        Tests that vehicles are only placed in edges listed in the
        edges_distribution parameter, when edges are specified
        """
        # set the initial_config parameters with an edges_distribution term for
        # only a few edges
        edges = ["top", "bottom"]
        initial_config = InitialConfig(edges_distribution=edges)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that all vehicles are only placed in edges specified in the
        # edges_distribution term
        for veh_id in self.env.vehicles.get_ids():
            if self.env.vehicles.get_edge(veh_id) not in edges:
                raise AssertionError

    def test_num_vehicles(self):
        """
        Tests that the number of starting positions generated is:
        - the number of vehicles in the vehicles class is no "num_vehicles"
          parameter is specified
        - equal to "num_vehicles" if this value is specified
        """
        # create the environment
        self.setUp_gen_start_pos()

        # check when "num_vehicles" is not specified
        startpos, startlanes, startspeeds = \
            self.env.scenario.generate_starting_positions()
        self.assertEqual(
            len(startpos), self.env.scenario.vehicles.num_vehicles)
        self.assertEqual(
            len(startlanes), self.env.scenario.vehicles.num_vehicles)

        # check when "num_vehicles" is specified
        startpos, startlanes, startspeeds = \
            self.env.scenario.generate_starting_positions(num_vehicles=10)
        self.assertEqual(len(startpos), 10)
        self.assertEqual(len(startlanes), 10)


class TestEvenStartPosInternalLinks(unittest.TestCase):
    """
    Tests the function gen_even_start_pos when internal links are being used.
    Ensures that all vehicles are evenly spaced except when a vehicle is
    supposed to be placed at an internal link, in which case the vehicle is
    placed right outside the internal link.
    """

    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=15)

        initial_config = InitialConfig(x0=150)

        # create the environment and scenario classes for a ring road
        self.env, scenario = figure_eight_exp_setup(
            initial_config=initial_config, vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_even_start_pos_internal(self):
        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(
            np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
            self.env.scenario.length)

        try:
            # if all element are equal, there should only be one unique value
            self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)
        except AssertionError:
            # check that, if not all vehicles are equally spaced, that the
            # vehicle that is not equally spaced is right after an internal
            # link, and at position 0
            for i in range(len(nth_headway) - 1):
                if nth_headway[i] - np.mean(nth_headway) > 0.001:
                    # if not, check that the last or first vehicle is right
                    # after an internal link, on position 0
                    pos = [
                        self.env.get_x_by_id(veh_id)
                        for veh_id in [ids[i + 1], ids[i]]
                    ]
                    rel_pos = [
                        self.env.scenario.get_edge(pos_i)[1] for pos_i in pos
                    ]

                    self.assertTrue(np.any(np.array(rel_pos) == 0))


class TestRandomStartPos(unittest.TestCase):
    """
    Tests the function gen_random_start_pos in base_scenario.py.
    """

    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        # ensures that the random starting position method is being used
        initial_config.spacing = "random"

        # create a multi-lane ring road network
        additional_net_params = {
            "length": 230,
            "lanes": 4,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(
            net_params=net_params,
            initial_config=initial_config,
            vehicles=vehicles)

    def tearDown_gen_start_pos(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_lanes_distribution(self):
        """
        Tests that vehicles are only placed in the requested number of lanes.
        """
        # create the environment
        initial_config = InitialConfig(spacing="random", lanes_distribution=2)
        self.setUp_gen_start_pos(initial_config)

        # verify that all vehicles are located in the number of allocated lanes
        for veh_id in self.env.vehicles.get_ids():
            if self.env.vehicles.get_lane(veh_id) >= \
                    initial_config.lanes_distribution:
                raise AssertionError

    def test_edges_distribution(self):
        """
        Tests that vehicles are only placed in the requested edges.
        """
        # set the initial_config parameters with an edges_distribution term for
        # only a few edges
        edges = ["top", "bottom"]
        initial_config = InitialConfig(
            spacing="random", edges_distribution=edges)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # check that all vehicles are only placed in edges specified in the
        # edges_distribution term
        for veh_id in self.env.vehicles.get_ids():
            if self.env.vehicles.get_edge(veh_id) not in edges:
                raise AssertionError


class TestEvenStartPosVariableLanes(unittest.TestCase):
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=50)

        initial_config = InitialConfig(lanes_distribution=5)

        # create the environment and scenario classes for a variable lanes per
        # edge ring road
        self.env, scenario = variable_lanes_exp_setup(
            vehicles=vehicles, initial_config=initial_config)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_even_start_pos_coverage(self):
        """
        Ensure that enough vehicles are placed in the network, and they cover
        all possible lanes.
        """
        expected_num_vehicles = self.env.vehicles.num_vehicles
        actual_num_vehicles = \
            len(self.env.traci_connection.vehicle.getIDList())

        # check that enough vehicles are in the network
        self.assertEqual(expected_num_vehicles, actual_num_vehicles)

        # check that all possible lanes are covered
        lanes = self.env.vehicles.get_lane(self.env.vehicles.get_ids())
        if any(i not in lanes for i in range(4)):
            raise AssertionError


class TestRandomStartPosVariableLanes(TestEvenStartPosVariableLanes):
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=50)

        initial_config = InitialConfig(spacing="random", lanes_distribution=5)

        # create the environment and scenario classes for a variable lanes per
        # edge ring road
        self.env, scenario = variable_lanes_exp_setup(
            vehicles=vehicles, initial_config=initial_config)


class TestEdgeLength(unittest.TestCase):
    """
    Tests the edge_length() method in the base scenario class.
    """

    def test_edge_length_edges(self):
        """
        Tests the edge_length() method when called on edges
        """
        additional_net_params = {
            "length": 1000,
            "lanes": 2,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, scenario = ring_road_exp_setup(net_params=net_params)

        self.assertEqual(scenario.edge_length("top"), 250)

    def test_edge_length_junctions(self):
        """
        Tests the speed_limit() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 1,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

        env, scenario = figure_eight_exp_setup(net_params=net_params)

        self.assertAlmostEqual(
            scenario.edge_length(":center_intersection_0"), 5.00)
        self.assertAlmostEqual(
            scenario.edge_length(":center_intersection_1"), 6.20)


class TestSpeedLimit(unittest.TestCase):
    """
    Tests the speed_limit() method in the base scenario class.
    """

    def test_speed_limit_edges(self):
        """
        Tests the speed_limit() method when called on edges
        """
        additional_net_params = {
            "length": 230,
            "lanes": 2,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, scenario = ring_road_exp_setup(net_params=net_params)

        self.assertAlmostEqual(scenario.speed_limit("top"), 60)

    def test_speed_limit_junctions(self):
        """
        Tests the speed_limit() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 1,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

        env, scenario = figure_eight_exp_setup(net_params=net_params)

        self.assertAlmostEqual(
            scenario.speed_limit("bottom_upper_ring_in"), 60)
        self.assertAlmostEqual(scenario.speed_limit(":top_upper_ring_0"), 60)


class TestNumLanes(unittest.TestCase):
    """
    Tests the num_lanes() method in the base scenario class.
    """

    def test_num_lanes_edges(self):
        """
        Tests the num_lanes() method when called on edges
        """
        additional_net_params = {
            "length": 230,
            "lanes": 2,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, scenario = ring_road_exp_setup(net_params=net_params)

        self.assertEqual(scenario.num_lanes("top"), 2)

    def test_num_lanes_junctions(self):
        """
        Tests the num_lanes() method when called on junctions
        """
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 3,
            "speed_limit": 60,
            "resolution": 40
        }
        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

        env, scenario = figure_eight_exp_setup(net_params=net_params)

        self.assertEqual(scenario.num_lanes("bottom_upper_ring_in"), 3)
        self.assertEqual(scenario.num_lanes(":top_upper_ring_0"), 3)


class TestGetEdgeList(unittest.TestCase):
    """
    Tests that the get_edge_list() in the scenario class properly returns all
    edges, and not junctions.
    """

    def setUp(self):
        # create the environment and scenario classes for a figure eight
        env, self.scenario = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def test_get_edge_list(self):
        edge_list = self.scenario.get_edge_list()
        expected_edge_list = [
            "bottom_lower_ring", "right_lower_ring_in", "right_lower_ring_out",
            "left_upper_ring", "top_upper_ring", "right_upper_ring",
            "bottom_upper_ring_in", "bottom_upper_ring_out", "top_lower_ring",
            "left_lower_ring"
        ]

        self.assertCountEqual(edge_list, expected_edge_list)


class TestGetJunctionList(unittest.TestCase):
    """
    Tests that the get_junction_list() in the scenario class properly returns
    all junctions, and no edges.
    """

    def setUp(self):
        # create the environment and scenario classes for a figure eight
        env, self.scenario = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def test_get_junction_list(self):
        junction_list = self.scenario.get_junction_list()
        expected_junction_list = \
            [':right_upper_ring_0', ':right_lower_ring_in_0',
             ':center_intersection_1', ':bottom_upper_ring_in_0',
             ':bottom_lower_ring_0', ':top_lower_ring_0',
             ':top_upper_ring_0', ':left_lower_ring_0',
             ':center_intersection_2', ':center_intersection_0',
             ':center_intersection_3', ':left_upper_ring_0']

        self.assertCountEqual(junction_list, expected_junction_list)


class TestNextPrevEdge(unittest.TestCase):
    """
    Tests that the next_edge() and prev_edge() methods returns the correct list
    of edges/lanes when looking to a scenario. This also tests that junctions
    are provided as next edges if they are before the next edge (e.g. a via to
    the next edge)
    """

    def test_next_edge_internal_links(self):
        """
        Tests the next_edge() method in the presence of internal links.
        """
        env, scenario = figure_eight_exp_setup()
        next_edge = scenario.next_edge("bottom_upper_ring_in", 0)
        expected_next_edge = [(':center_intersection_0', 0),
                              (':center_intersection_1', 0)]

        self.assertCountEqual(next_edge, expected_next_edge)

    def test_prev_edge_internal_links(self):
        """
        Tests the prev_edge() method in the presence of internal links.
        """
        env, scenario = figure_eight_exp_setup()
        prev_edge = scenario.prev_edge("bottom_upper_ring_in", 0)
        expected_prev_edge = [(':bottom_upper_ring_in_0', 0)]

        self.assertCountEqual(prev_edge, expected_prev_edge)

    def test_next_edge_no_internal_links(self):
        """
        Tests the next_edge() method in the absence of internal links.
        """
        env, scenario = ring_road_exp_setup()
        next_edge = scenario.next_edge("top", 0)
        expected_next_edge = [("left", 0)]

        self.assertCountEqual(next_edge, expected_next_edge)

    def test_prev_edge_no_internal_links(self):
        """
        Tests the prev_edge() method in the absence of internal links.
        """
        env, scenario = ring_road_exp_setup()
        prev_edge = scenario.prev_edge("top", 0)
        expected_prev_edge = [("right", 0)]

        self.assertCountEqual(prev_edge, expected_prev_edge)

    def test_no_edge_ahead(self):
        """
        Tests that, when there are no edges in front, next_edge() returns an
        empty list
        """
        env, scenario = highway_exp_setup()
        next_edge = scenario.next_edge(env.scenario.get_edge_list()[0], 0)
        self.assertTrue(len(next_edge) == 0)

    def test_no_edge_behind(self):
        """
        Tests that, when there are no edges behind, prev_edge() returns an
        empty list
        """
        env, scenario = highway_exp_setup()
        prev_edge = scenario.prev_edge(env.scenario.get_edge_list()[0], 0)
        self.assertTrue(len(prev_edge) == 0)


if __name__ == '__main__':
    unittest.main()
