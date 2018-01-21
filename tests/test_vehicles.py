import unittest
from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import IDMController


class TestVehiclesClass(unittest.TestCase):
    """
    Tests various functions in the vehicles class
     - Checks to make sure vehicle class correctly specifies lane change and speed modes
    """
    def runTest(self):
        vehicles = Vehicles()
        vehicles.add_vehicles("typeA",
                    (IDMController, {}),
                    speed_mode = 'no_collide',
                    lane_change_mode = "no_lat_collide")

        self.assertEqual(vehicles.get_speed_mode_name("typeA_0"), "no_collide")
        self.assertEqual(vehicles.get_speed_mode("typeA_0"), 31)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeA_0"), "no_lat_collide")
        self.assertEqual(vehicles.get_lane_change_mode("typeA_0"), 256)

        vehicles.add_vehicles("typeB",
                    (IDMController, {}),
                    speed_mode = 'aggressive',
                    lane_change_mode = "strategic")

        self.assertEqual(vehicles.get_speed_mode_name("typeB_0"), "aggressive")
        self.assertEqual(vehicles.get_speed_mode("typeB_0"), 0)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeB_0"), "strategic")
        self.assertEqual(vehicles.get_lane_change_mode("typeB_0"), 853)

        vehicles.add_vehicles("typeC",
                    (IDMController, {}),
                    speed_mode = 'custom',
                    custom_speed_mode = 16,
                    lane_change_mode = "custom",
                    custom_lane_change_mode=277)
        self.assertEqual(vehicles.get_speed_mode_name("typeC_0"), "custom")
        self.assertEqual(vehicles.get_speed_mode("typeC_0"), 16)
        self.assertEqual(vehicles.get_lane_change_mode_name("typeC_0"), "custom")
        self.assertEqual(vehicles.get_lane_change_mode("typeC_0"), 277)


if __name__ == '__main__':
    unittest.main()