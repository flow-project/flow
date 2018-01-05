import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import *

from tests.setup_scripts import ring_road_exp_setup, figure_eight_exp_setup
from tests.setup_scripts import variable_lanes_exp_setup


class TestGetX(unittest.TestCase):
    """
    Tests the get_x function for vehicles placed in links and in junctions. This
    is tested on a scenario whose edgestarts are known beforehand (figure 8).
    """
    def setUp(self):
        # create the environment and scenario classes for a figure eight
        env, self.scenario = figure_eight_exp_setup()

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def runTest(self):
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

    def runTest(self):
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
    Tests the function gen_even_start_pos in base_scenario.py. This function can
    be used on any scenario subclass, and therefore may be tested on any of
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
        additional_net_params = {"length": 230, "lanes": 4, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=15)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
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
        initial_config = InitialConfig()

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
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
        initial_config = InitialConfig(x0=x0)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
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
        initial_config = InitialConfig(bunching=bunching)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
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
        initial_config = InitialConfig(bunching=bunching)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
                             self.env.scenario.length)

        # check that all vehicles, including the last vehicle, have the same
        # spacing
        self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_lanes_distribution(self):
        """
        Tests that if lanes_distribution is less than the total number of lanes,
        the vehicles are uniformly distributed over the specified number of
        lanes.
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
            veh_pos.append([self.env.get_x_by_id(veh_id)
                            for veh_id in ids
                            if self.env.vehicles.get_lane(veh_id) == i])

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
        Tests that when lanes_distribution is less than 1, the number is set to 1
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
        # set the initial_config parameters with a very large lanes_distribution
        lanes_distribution = np.inf
        initial_config = InitialConfig(lanes_distribution=lanes_distribution)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = []
        for i in range(self.env.scenario.lanes):
            veh_pos.append([self.env.get_x_by_id(veh_id)
                            for veh_id in ids
                            if self.env.vehicles.get_lane(veh_id) == i])

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
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=15)

        initial_config = InitialConfig(x0=150)

        # create the environment and scenario classes for a ring road
        self.env, scenario = figure_eight_exp_setup(
            initial_config=initial_config,
            vehicles=vehicles
        )

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        veh_pos = np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # difference in position between the nth vehicle and the vehicle ahead
        # of it
        nth_headway = np.mod(np.append(veh_pos[1:], veh_pos[0]) - veh_pos,
                             self.env.scenario.length)

        try:
            # if all element are equal, there should only be one unique value
            self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)
        except AssertionError:
            # check that, if not all vehicles are equally spaced, that the
            # vehicle that is not equally spaced is right after an internal
            # link, and at position 0
            for i in range(len(nth_headway)-1):
                if nth_headway[i] - np.mean(nth_headway) > 0.001:
                    # if not, check that the last or first vehicle is right
                    # after an internal link, on position 0
                    pos = [self.env.get_x_by_id(veh_id) for veh_id in
                           [ids[i + 1], ids[i]]]
                    rel_pos = [self.env.scenario.get_edge(pos_i)[1] for pos_i
                               in pos]

                    self.assertTrue(np.any(np.array(rel_pos) == 0))


class TestRandomStartPos(unittest.TestCase):
    """
    Tests the function gen_random_start_pos in base_scenario.py.
    """
    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        # ensures that the random starting position method is being used
        initial_config.spacing = "random"

        # create a multi-lane ring road network
        additional_net_params = {"length": 230, "lanes": 4, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
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
        initial_config = InitialConfig(spacing="random",
                                       edges_distribution=edges)

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
        vehicles.add(veh_id="test",
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

    def runTest(self):
        """
        Ensures that enough vehicles are placed in the network, and they cover
        all possible lanes.
        """
        expected_num_vehicles = self.env.vehicles.num_vehicles
        actual_num_vehicles = len(self.env.traci_connection.vehicle.getIDList())

        # check that enough vehicles are in the network
        self.assertEqual(expected_num_vehicles, actual_num_vehicles)

        # check that all possible lanes are covered
        lanes = self.env.vehicles.get_lane()
        if any([i not in lanes for i in range(4)]):
            raise AssertionError


class TestRandomStartPosVariableLanes(TestEvenStartPosVariableLanes):
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
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
    def setUp(self):
        additional_net_params = {"length": 1000, "lanes": 2,
                                 "speed_limit": 60, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, self.scenario = ring_road_exp_setup(net_params=net_params)

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def runTest(self):
        self.assertEqual(self.scenario.edge_length("top"), 250)


class TestSpeedLimit(unittest.TestCase):
    """
    Tests the speed_limit() method in the base scenario class.
    """
    def setUp(self):
        additional_net_params = {"length": 230, "lanes": 2,
                                 "speed_limit": 60, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, self.scenario = ring_road_exp_setup(net_params=net_params)

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def runTest(self):
        self.assertEqual(int(self.scenario.speed_limit("top")), 60)


class TestNumLanes(unittest.TestCase):
    """
    Tests the num_lanes() method in the base scenario class.
    """
    def setUp(self):
        additional_net_params = {"length": 230, "lanes": 2,
                                 "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a figure eight
        env, self.scenario = ring_road_exp_setup(net_params=net_params)

    def tearDown(self):
        # free data used by the class
        self.scenario = None

    def runTest(self):
        self.assertEqual(self.scenario.num_lanes("top"), 2)


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

    def runTest(self):
        edge_list = self.scenario.get_edge_list()
        expected_edge_list = ["bottom_lower_ring", "right_lower_ring_in",
                              "right_lower_ring_out", "left_upper_ring",
                              "top_upper_ring", "right_upper_ring",
                              "bottom_upper_ring_in", "bottom_upper_ring_out",
                              "top_lower_ring", "left_lower_ring"]

        self.assertCountEqual(edge_list, expected_edge_list)


if __name__ == '__main__':
    unittest.main()
