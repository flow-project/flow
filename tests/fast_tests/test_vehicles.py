import unittest
from flow.core.vehicles import Vehicles
from flow.core.params import SumoCarFollowingParams
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController


class TestVehiclesClass(unittest.TestCase):
    """
    Tests various functions in the vehicles class
    """
    def runSpeedLaneChangeModes(self):
        """
        Checks to make sure vehicle class correctly specifies lane change and
        speed modes
        """
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")

        self.assertEqual(vehicles.get_speed_mode("typeA_0"), 1)
        self.assertEqual(vehicles.get_lane_change_mode("typeA_0"), 256)

        vehicles.add("typeB",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='aggressive',
                     lane_change_mode="strategic")

        self.assertEqual(vehicles.get_speed_mode("typeB_0"), 0)
        self.assertEqual(vehicles.get_lane_change_mode("typeB_0"), 853)

        vehicles.add("typeC",
                     acceleration_controller=(IDMController, {}),
                     speed_mode=31,
                     lane_change_mode=277)
        self.assertEqual(vehicles.get_speed_mode("typeC_0"), 31)
        self.assertEqual(vehicles.get_lane_change_mode("typeC_0"), 277)

    def test_controlled_id_params(self):
        """
        Ensures that, if a vehicle is not a sumo vehicle, then minGap is set to
        zero so that all headway values are correct.
        """
        # check that, if the vehicle is not a SumoCarFollowingController
        # vehicle, then its minGap is equal to 0
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")
        self.assertEqual(vehicles.types[0][1]["minGap"], 0)

        # check that, if the vehicle is a SumoCarFollowingController vehicle,
        # then its minGap, accel, and decel are set to default
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(SumoCarFollowingController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")
        default_minGap = SumoCarFollowingParams().controller_params["minGap"]
        self.assertEqual(vehicles.types[0][1]["minGap"], default_minGap)

    def test_add_vehicles_human(self):
        """
        Ensures that added human vehicles are placed in the current vehicle IDs,
        and that the number of vehicles is correct.
        """
        # generate a vehicles class
        vehicles = Vehicles()

        # vehicles whose acceleration and LC are controlled by sumo
        vehicles.add("test_1", num_vehicles=1)

        # vehicles whose acceleration are controlled by sumo
        vehicles.add("test_2", num_vehicles=2,
                     lane_change_controller=(StaticLaneChanger, {}))

        # vehicles whose LC are controlled by sumo
        vehicles.add("test_3", num_vehicles=4,
                     acceleration_controller=(IDMController, {}))

        self.assertEqual(vehicles.num_vehicles, 7)
        self.assertEqual(len(vehicles.get_ids()), 7)
        self.assertEqual(len(vehicles.get_rl_ids()), 0)
        self.assertEqual(len(vehicles.get_human_ids()), 7)
        self.assertEqual(len(vehicles.get_controlled_ids()), 4)
        self.assertEqual(len(vehicles.get_controlled_lc_ids()), 2)

    def test_add_vehicles_rl(self):
        """
        Ensures that added rl vehicles are placed in the current vehicle IDs,
        and that the number of vehicles is correct.
        """
        vehicles = Vehicles()
        vehicles.add("test_rl", num_vehicles=10,
                     acceleration_controller=(RLCarFollowingController, {}))

        self.assertEqual(vehicles.num_vehicles, 10)
        self.assertEqual(len(vehicles.get_ids()), 10)
        self.assertEqual(len(vehicles.get_rl_ids()), 10)
        self.assertEqual(len(vehicles.get_human_ids()), 0)
        self.assertEqual(len(vehicles.get_controlled_ids()), 0)
        self.assertEqual(len(vehicles.get_controlled_lc_ids()), 0)

    def test_remove(self):
        """
        Checks that there is no trace of the vehicle ID of the vehicle meant to
        be removed in the vehicles class.
        """
        # generate a vehicles class
        vehicles = Vehicles()
        vehicles.add("test", None, num_vehicles=10)
        vehicles.add("test_rl", num_vehicles=10,
                     acceleration_controller=(RLCarFollowingController, {}))

        # remove one human-driven vehicle and on rl vehicle
        vehicles.remove("test_0")
        vehicles.remove("test_rl_0")

        # ensure that the removed vehicle's ID is not in any lists of vehicles
        if "test_0" in vehicles.get_ids():
            raise AssertionError("vehicle still in get_ids()")
        if "test_0" in vehicles.get_human_ids():
            raise AssertionError("vehicle still in get_controlled_lc_ids()")
        if "test_0" in vehicles.get_controlled_lc_ids():
            raise AssertionError("vehicle still in get_controlled_lc_ids()")
        if "test_0" in vehicles.get_controlled_ids():
            raise AssertionError("vehicle still in get_controlled_ids()")
        if "test_rl_0" in vehicles.get_ids():
            raise AssertionError("RL vehicle still in get_ids()")
        if "test_rl_0" in vehicles.get_rl_ids():
            raise AssertionError("RL vehicle still in get_rl_ids()")

        # ensure that the vehicles are not storing extra information in the
        # vehicles.__vehicles dict
        self.assertRaises(KeyError, vehicles.get_state, 'test_0', "type")
        self.assertRaises(KeyError, vehicles.get_state, "rl_test_0", "type")

        # ensure that the num_vehicles matches the actual number of vehicles
        self.assertEqual(vehicles.num_vehicles, len(vehicles.get_ids()))

        # ensures that then num_rl_vehicles matches the actual number of rl veh
        self.assertEqual(vehicles.num_rl_vehicles, len(vehicles.get_rl_ids()))


if __name__ == '__main__':
    unittest.main()
