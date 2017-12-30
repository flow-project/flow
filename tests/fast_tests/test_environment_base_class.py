import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import *

from tests.setup_scripts import ring_road_exp_setup


class TestStartingPositionShuffle(unittest.TestCase):
    """
    Tests that, at resets, the starting position of vehicles changes while
    keeping the ordering and relative spacing between vehicles.
    """
    def setUp(self):
        # turn on starting position shuffle
        env_params = EnvParams(starting_position_shuffle=True,
                               additional_params={"target_velocity": 30})

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=5)

        initial_config = InitialConfig(x0=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        ids = self.env.vehicles.get_ids()

        # position of vehicles before reset
        before_reset = \
            np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # reset the environment
        self.env.reset()

        # position of vehicles after reset
        after_reset = \
            np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        offset = after_reset[0] - before_reset[0]

        # remove the new offset from the original positions after reset
        after_reset = np.mod(after_reset - offset, self.env.scenario.length)

        np.testing.assert_array_almost_equal(before_reset, after_reset)


class TestVehicleArrangementShuffle(unittest.TestCase):
    """
    Tests that, at resets, the ordering of vehicles changes while the starting
    position values stay the same.
    """
    def setUp(self):
        # turn on vehicle arrangement shuffle
        env_params = EnvParams(vehicle_arrangement_shuffle=True,
                               additional_params={"target_velocity": 30})

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=5)

        initial_config = InitialConfig(x0=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        ids = self.env.vehicles.get_ids()

        # position of vehicles before reset
        before_reset = [self.env.get_x_by_id(veh_id) for veh_id in ids]

        # reset the environment
        self.env.reset()

        # position of vehicles after reset
        after_reset = [self.env.get_x_by_id(veh_id) for veh_id in ids]

        self.assertCountEqual(before_reset, after_reset)


class TestEmissionPath(unittest.TestCase):
    """
    Tests that the default emission path of an environment is set to None. If it
    is not None, then sumo starts accumulating memory.
    """
    def setUp(self):
        # set sumo_params to default
        sumo_params = SumoParams()

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(sumo_params=sumo_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def runTest(self):
        self.assertIsNone(self.env.emission_path)


# class TestResetAfterVehicleCollision(unittest.TestCase):
#     """
#     Tests that colliding vehicles are properly introduced back into the network
#     during reset.
#     """
#     def setUp(self):
#         # TODO: figure out when it occurs exactly (its not always), and what
#         # TODO: the exact fix is
#         pass
#
#     def tearDown(self):
#         pass
#
#     def runTest(self):
#         pass


class TestApplyingActionsWithSumo(unittest.TestCase):
    """
    Tests the apply_acceleration, apply_lane_change, and choose_routes functions
    in base_env.py
    """
    def setUp(self):
        # create a 2-lane ring road network
        additional_net_params = {"length": 230, "lanes": 3, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # turn on starting position shuffle
        env_params = EnvParams(starting_position_shuffle=True,
                               additional_params={"target_velocity": 30})

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add_vehicles(veh_id="test",
                              acceleration_controller=(IDMController, {}),
                              routing_controller=(ContinuousRouter, {}),
                              num_vehicles=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 env_params=env_params,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_apply_acceleration(self):
        """
        Tests that, in the absence of all failsafes, the acceleration requested
        from sumo is equal to the acceleration witnessed in between steps. Also
        ensures that vehicles can never have velocities below zero given any
        acceleration.
        """
        ids = self.env.vehicles.get_ids()

        vel0 = np.array([self.env.vehicles.get_speed(veh_id) for veh_id in ids])

        # apply a certain set of accelerations to the vehicles in the network
        accel_step0 = np.array([0, 1, 4, 9, 16])
        self.env.apply_acceleration(veh_ids=ids, acc=accel_step0)
        self.env.traci_connection.simulationStep()

        # compare the new velocity of the vehicles to the expected velocity
        # given the accelerations
        vel1 = np.array([self.env.traci_connection.vehicle.getSpeed(veh_id)
                         for veh_id in ids])
        expected_vel1 = (vel0 + accel_step0 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel1, expected_vel1, 1)

        # collect new network observations from sumo
        network_observations = \
            self.env.traci_connection.vehicle.getSubscriptionResults()

        # get the list of vehicles currently in the network
        id_list = self.env.traci_connection.vehicle.getIDList()

        # store the network observations in the vehicles class
        self.env.vehicles.set_sumo_observations(network_observations, id_list,
                                                self.env)

        # apply a set of decelerations
        accel_step1 = np.array([-16, -9, -4, -1, 0])
        self.env.apply_acceleration(veh_ids=ids, acc=accel_step1)
        self.env.traci_connection.simulationStep()

        # this time, some vehicles should be at 0 velocity (NOT less), and sum
        # are a result of the accelerations that took place
        vel2 = np.array([self.env.traci_connection.vehicle.getSpeed(veh_id)
                         for veh_id in ids])
        expected_vel2 = (vel1 + accel_step1 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel2, expected_vel2, 1)

    def test_apply_lane_change_errors(self):
        """
        Ensures that apply_lane_change raises ValueErrors when it should
        """
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        # make sure that running apply lane change with a invalid direction
        # values leads to a ValueError
        bad_directions = np.array([-1, 0, 1, 2, 3])

        self.assertRaises(
            ValueError,
            self.env.apply_lane_change, veh_ids=ids, direction=bad_directions)

        # make sure that running apply_lane_change with both directions and
        # target_lames leads to a ValueError
        self.assertRaises(
            ValueError,
            self.env.apply_lane_change,
            veh_ids=ids, direction=[], target_lane=[])

    def test_apply_lane_change_direction(self):
        """
        Tests the direction method for apply_lane_change. Ensures that the lane
        change action requested from sumo is the same as the lane change that
        occurs, and that vehicles attempting do not issue lane changes in there
        is no lane in te requested direction.
        """
        self.env.reset()
        ids = self.env.vehicles.get_ids()
        lane0 = np.array([self.env.vehicles.get_lane(veh_id) for veh_id in ids])

        # perform lane-changing actions using the direction method
        direction0 = np.array([0, 1, 0, 1, -1])
        self.env.apply_lane_change(ids, direction=direction0)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane1 = np.array([self.env.traci_connection.vehicle.getLaneIndex(veh_id)
                          for veh_id in ids])
        expected_lane1 = (lane0 + np.sign(direction0)).clip(
            min=0, max=self.env.scenario.lanes-1)

        np.testing.assert_array_almost_equal(lane1, expected_lane1, 1)

        # collect new network observations from sumo
        network_observations = \
            self.env.traci_connection.vehicle.getSubscriptionResults()

        # get the list of vehicles currently in the network
        id_list = self.env.traci_connection.vehicle.getIDList()

        # store the network observations in the vehicles class
        self.env.vehicles.set_sumo_observations(network_observations, id_list,
                                                self.env)

        # perform lane-changing actions using the direction method one more
        # time to test lane changes to the right
        direction1 = np.array([-1, -1, -1, -1, -1])
        self.env.apply_lane_change(ids, direction=direction1)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane2 = np.array([self.env.traci_connection.vehicle.getLaneIndex(veh_id)
                          for veh_id in ids])
        expected_lane2 = (lane1 + np.sign(direction1)).clip(
            min=0, max=self.env.scenario.lanes-1)

        np.testing.assert_array_almost_equal(lane2, expected_lane2, 1)

    def test_apply_lane_change_target_lane(self):
        """
        Tests the target_lane method for apply_lane_change. Ensure that vehicles
        do not jump multiple lanes in a single step, and that invalid directions
        do not lead to errors with sumo.
        """
        self.env.reset()
        ids = self.env.vehicles.get_ids()
        lane0 = np.array([self.env.vehicles.get_lane(veh_id) for veh_id in ids])

        # perform lane-changing actions using the direction method
        target_lane0 = np.array([0, 1, 2, 1, -1])
        self.env.apply_lane_change(ids, target_lane=target_lane0)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane1 = np.array([self.env.traci_connection.vehicle.getLaneIndex(veh_id)
                          for veh_id in ids])
        expected_lane1 = (lane0 + np.sign(target_lane0 - lane0)).clip(
            min=0, max=self.env.scenario.lanes-1)

        np.testing.assert_array_almost_equal(lane1, expected_lane1, 1)

        # collect new network observations from sumo
        network_observations = \
            self.env.traci_connection.vehicle.getSubscriptionResults()

        # get the list of vehicles currently in the network
        id_list = self.env.traci_connection.vehicle.getIDList()

        # store the network observations in the vehicles class
        self.env.vehicles.set_sumo_observations(network_observations, id_list,
                                                self.env)

        # perform lane-changing actions using the direction method one more
        # time to test lane changes to the right
        target_lane1 = np.array([-1, -1, 2, -1, -1])
        self.env.apply_lane_change(ids, target_lane=target_lane1)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane2 = np.array([self.env.traci_connection.vehicle.getLaneIndex(veh_id)
                          for veh_id in ids])
        expected_lane2 = (lane1 + np.sign(target_lane1 - lane1)).clip(
            min=0, max=self.env.scenario.lanes-1)

        np.testing.assert_array_almost_equal(lane2, expected_lane2, 1)

    def test_choose_route(self):
        """
        Tests that:
        - when no route changing action is requested, it is skipped
        - if a route change action is requested, the route of the vehicle in the
          new step is the same of the route requested
        """
        self.env.reset()

        ids = self.env.vehicles.get_ids()

        # collect the original routes of all vehicles
        routes_0 = [self.env.traci_connection.vehicle.getRoute(veh_id)
                    for veh_id in ids]

        # assign a new route of None to all vehicles
        self.env.choose_routes(ids, [None]*len(ids))
        self.env.traci_connection.simulationStep()

        # check that when vehicles return a route of run, no error is outputted,
        # and the routes of the vehicles remain the same
        routes_1 = [self.env.traci_connection.vehicle.getRoute(veh_id)
                    for veh_id in ids]

        self.assertSequenceEqual(routes_0, routes_1)

        # replace one of the routes with something new
        self.env.choose_routes(ids, [routes_1[1]] + [None]*(len(ids)-1))
        self.env.traci_connection.simulationStep()

        # check that when vehicles return a route of run, no error is outputted,
        # and the routes of the vehicles remain the same
        routes_2 = [self.env.traci_connection.vehicle.getRoute(veh_id)
                    for veh_id in ids]

        # check that the new route was changed
        self.assertSequenceEqual(routes_2[0], routes_1[1])


if __name__ == '__main__':
    unittest.main()
