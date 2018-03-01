import unittest

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, SumoCarFollowingParams, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.routing_controllers import ContinuousRouter, GridRouter

from tests.setup_scripts import grid_mxn_exp_setup


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # create the environment and scenario classes for a ring road
        Sumo_Params = SumoParams(sim_step=1,
                                 sumo_binary="sumo-gui")
        total_vehicles = 20
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(SumoCarFollowingController,
                                              {}),
                     routing_controller=(GridRouter, {}),
                     sumo_car_following_params=
                     SumoCarFollowingParams(tau=0.1, carFollowModel="Krauss", minGap=2.5),
                     num_vehicles=total_vehicles,
                     speed_mode=0b00000)
        total_vehicles = vehicles.num_vehicles
        grid_array = {"short_length": 100, "inner_length": 100,
                      "long_length": 100, "row_num": 1,
                      "col_num": 1,
                      "cars_left": int(total_vehicles / 4),
                      "cars_right": int(total_vehicles / 4),
                      "cars_top": int(total_vehicles / 4),
                      "cars_bot": int(total_vehicles / 4)}

        additional_net_params = {"length": 200, "lanes": 2, "speed_limit": 35,
                                 "resolution": 40, "grid_array": grid_array,
                                 "horizontal_lanes": 1, "vertical_lanes": 1,
                                 "traffic_lights": 1}

        net_params = NetParams(no_internal_links=False,
                               additional_params=additional_net_params)

        self.env, self.scenario = grid_mxn_exp_setup(row_num=1, col_num=1,
                                                     sumo_params=Sumo_Params,
                                                     vehicles=vehicles,
                                                     net_params=net_params)

        # go through the env and set all the lights to green
        for i in range(self.env.rows * self.env.cols):
            self.env.traci_connection.trafficlights.setRedYellowGreenState(
                'center' + str(i), "gggggggggggg")

        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, self.scenario)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free up used memory
        self.env = None
        self.exp = None

    def test_collide(self):
        self.exp.run(1, 50)

if __name__ == '__main__':
    unittest.main()