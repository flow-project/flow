import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import *

from tests.setup_scripts import ring_road_exp_setup, figure_eight_exp_setup


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
        vehicles.add_vehicles(veh_id="test",
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

    def test_distribution_length(self):
        """
        Tests that vehicles are uniformly distributed given a certain
        distribution length
        """
        # set the initial_config parameters with a distribution length in
        # between zero and the length of the network
        distribution_length = 150
        initial_config = InitialConfig(distribution_length=distribution_length)

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
        # difference between the distribution length and the length of the
        # network
        self.assertAlmostEqual(nth_headway[-1] - nth_headway[-2],
                               self.env.scenario.length - distribution_length)

        # delete the created environment
        self.tearDown_gen_start_pos()

    def test_distribution_length_too_large(self):
        """
        Tests that if the distribution_length is greater than the length of the
        network, the distribution length is set to the length of the network
        instead
        """
        # set the initial_config parameters with a very large distribution
        # length
        distribution_length = np.inf
        initial_config = InitialConfig(distribution_length=distribution_length)

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

    def test_bunching_distribution_length_too_compact(self):
        """
        Tests that if bunching is too large or the distribution length is too
        small, the vehicles are bunched as close to each other as possible
        """
        # set the initial_config parameters with a very large bunching term
        bunching = np.inf
        initial_config = InitialConfig(bunching=bunching)

        # create the environment
        self.setUp_gen_start_pos(initial_config)

        # get the positions of all vehicles
        ids = self.env.vehicles.get_ids()
        headway = [self.env.vehicles.get_headway(veh_id) for veh_id in ids]

        # check that all headways (except that of the front vehicle) are zero
        self.assertEqual(sum(headway[:-1]), 0)

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
        # requested lanes lanes
        for i in range(lanes_distribution):
            # difference in position between the nth vehicle and the vehicle
            # ahead of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.scenario.length)

            # if all element are equal, there should only be one unique value
            if i >= lanes_distribution - \
                    self.env.vehicles.num_vehicles % lanes_distribution:
                # in the case of an odd number of vehicles, the second
                self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)
            else:
                self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

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
            # difference in position between the nth vehicle and the vehicle ahead
            # of it
            nth_headway = \
                np.mod(np.append(veh_pos[i][1:], veh_pos[i][0]) - veh_pos[i],
                       self.env.scenario.length)

            # if all element are equal, there should only be one unique value
            if i >= self.env.scenario.lanes - \
                    self.env.vehicles.num_vehicles % self.env.scenario.lanes:
                # in the case of an odd number of vehicles, the second
                self.assertEqual(np.unique(np.around(nth_headway[:-1], 2)).size, 1)
            else:
                self.assertEqual(np.unique(np.around(nth_headway, 2)).size, 1)

        # delete the created environment
        self.tearDown_gen_start_pos()


class TestGaussianStartPos(TestEvenStartPos):
    """
    Tests the function gen_gaussian_start_pos in base_scenario.py. This function
    can be used on any scenario subclass, and therefore may be tested on any of
    these classes. In order to perform this testing, replace the scenario in
    setUp() with the scenario to be tested.
    """
    def setUp_gen_start_pos(self, initial_config=InitialConfig()):
        """
        Replace with any scenario you would like to test gen_gaussian_start_pos
        on. In ordering for all the tests to be meaningful, the scenario must
        contain MORE THAN TWO LANES.
        """
        # create a multi-lane ring road network
        additional_net_params = {"length": 230, "lanes": 4, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=15)

        # makes sure that all tests are being perform on a gaussian starting pos
        initial_config.spacing = "gaussian"
        if initial_config.scale == InitialConfig().scale:
            initial_config.scale = 0

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
        Ignore from parent class
        """
        pass

    def test_zero_scale(self):
        """
        Tests that get_gaussian_start_pos distributes vehicles evenly when
        scale is zero.
        """
        # set the initial_config parameters with an x0 value that is something
        # in between zero and the length of the network
        initial_config = InitialConfig(spacing="gaussian", scale=0)

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


class TestGaussianAdditiveStartPos(TestEvenStartPos):
    """
    Tests the function gen_gaussian_additive_start_pos in base_scenario.py. This
    function can be used on any scenario subclass, and therefore may be tested
    on any of these classes. In order to perform this testing, replace the
    scenario in setUp() with the scenario to be tested.
    """

    def setUp_gen_start_pos(
            self, initial_config=InitialConfig()):
        """
        Replace with any scenario you would like to test gen_gaussian_start_pos
        on. In ordering for all the tests to be meaningful, the scenario must
        contain MORE THAN TWO LANES.
        """
        # create a multi-lane ring road network
        additional_net_params = {"length": 230, "lanes": 4, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=15)

        initial_config.spacing = "gaussian_additive"
        if initial_config.downscale == InitialConfig().downscale:
            initial_config.downscale = np.inf

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
        Ignore from parent class
        """
        pass

    def test_infinite_downscale(self):
        """
        Tests that get_gaussian_additive_start_pos produces a uniform
        distribution when downscale is infinite.
        """
        # set the initial_config parameters with an x0 value that is something
        # in between zero and the length of the network
        initial_config = InitialConfig(spacing="gaussian_additive",
                                       downscale=np.inf)

        # create the environment, test that it can be created without failing
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
        vehicles.add_vehicles(veh_id="test",
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


class TestGaussianStartPosInternalLinks(TestEvenStartPosInternalLinks):
    """
    Tests the function gen_gaussian_start_pos when internal links are being
    used. Ensures that, if the scale is 0, all vehicles are evenly spaced except
    when a vehicle is supposed to be placed at an internal link, in which case
    the vehicle is placed right outside the internal link.
    """
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=15)

        initial_config = InitialConfig(spacing="gaussian", scale=0, x0=150)

        # create the environment and scenario classes for a ring road
        self.env, scenario = figure_eight_exp_setup(
            initial_config=initial_config,
            vehicles=vehicles
        )


class TestGaussianAdditiveStartPosInternalLinks(TestEvenStartPosInternalLinks):
    """
    Tests the function gen_gaussian_additive_start_pos when internal links are
    being used. Ensures that, if the scale is 0, all vehicles are evenly spaced
    except when a vehicle is supposed to be placed at an internal link, in which
    case the vehicle is placed right outside the internal link.
    """
    def setUp(self):
        # place 15 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=15)

        initial_config = InitialConfig(spacing="gaussian_additive",
                                       downscale=np.inf, x0=150)

        # create the environment and scenario classes for a ring road
        self.env, scenario = figure_eight_exp_setup(
            initial_config=initial_config,
            vehicles=vehicles
        )


if __name__ == '__main__':
    unittest.main()
