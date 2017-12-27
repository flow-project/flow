import unittest
from flow.core.vehicles import Vehicles
from flow.core.params import SumoCarFollowingParams
from flow.controllers.car_following_models import *


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
        vehicles.add_vehicles("typeA",
                    (IDMController, {}),
                    speed_mode='no_collide',
                    lane_change_mode="no_lat_collide")

        self.assertEqual(vehicles.get_speed_mode_name("typeA_0"), "no_collide")
        self.assertEqual(vehicles.get_speed_mode("typeA_0"), 1)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeA_0"), "no_lat_collide")
        self.assertEqual(vehicles.get_lane_change_mode("typeA_0"), 256)

        vehicles.add_vehicles("typeB",
                    (IDMController, {}),
                    speed_mode='aggressive',
                    lane_change_mode="strategic")

        self.assertEqual(vehicles.get_speed_mode_name("typeB_0"), "aggressive")
        self.assertEqual(vehicles.get_speed_mode("typeB_0"), 0)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeB_0"), "strategic")
        self.assertEqual(vehicles.get_lane_change_mode("typeB_0"), 853)

        vehicles.add_vehicles("typeC",
                    (IDMController, {}),
                    speed_mode='custom',
                    custom_speed_mode=31,
                    lane_change_mode="custom",
                    custom_lane_change_mode=277)
        self.assertEqual(vehicles.get_speed_mode_name("typeC_0"), "custom")
        self.assertEqual(vehicles.get_speed_mode("typeC_0"), 31)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeC_0"), "custom")
        self.assertEqual(vehicles.get_lane_change_mode("typeC_0"), 277)

    def test_controlled_id_params(self):
        """
        Ensures that, if a vehicle is not a sumo vehicle, its max acceleration /
        deceleration are set to very large values to allow fuller control, and
        that minGap is set to zero so that all headway values are correct.
        """
        # check that, if the vehicle is not a SumoCarFollowingController
        # vehicle, then its minGap is equal to 0, and its accel/decel is 100
        vehicles = Vehicles()
        vehicles.add_vehicles("typeA",
                    (IDMController, {}),
                    speed_mode='no_collide',
                    lane_change_mode="no_lat_collide")
        self.assertEqual(vehicles.types[0][1]["minGap"], 0)
        self.assertEqual(vehicles.types[0][1]["accel"], 100)
        self.assertEqual(vehicles.types[0][1]["decel"], 100)

        # check that, if the vehicle is a SumoCarFollowingController vehicle,
        # then its minGap, accel, and decel are set to default
        vehicles = Vehicles()
        vehicles.add_vehicles("typeA",
                    (SumoCarFollowingController, {}),
                    speed_mode='no_collide',
                    lane_change_mode="no_lat_collide")
        default_minGap = SumoCarFollowingParams().controller_params["minGap"]
        default_accel = SumoCarFollowingParams().controller_params["accel"]
        default_decel = SumoCarFollowingParams().controller_params["decel"]
        self.assertEqual(vehicles.types[0][1]["minGap"], default_minGap)
        self.assertEqual(vehicles.types[0][1]["accel"], default_accel)
        self.assertEqual(vehicles.types[0][1]["decel"], default_decel)

if __name__ == '__main__':
    unittest.main()
