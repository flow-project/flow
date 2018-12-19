import unittest

from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, SumoCarFollowingParams, NetParams, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.routing_controllers import GridRouter

from tests.setup_scripts import grid_mxn_exp_setup


class TestCollisions(unittest.TestCase):
    """Tests that collisions do not cause the experiments to terminate
    prematurely."""

    def test_collide(self):
        """Tests collisions in the absence of inflows."""
        # create the environment and scenario classes for a ring road
        sumo_params = SumoParams(sim_step=1, render=False)
        total_vehicles = 20
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(SumoCarFollowingController, {}),
            routing_controller=(GridRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                tau=0.1, carFollowModel="Krauss", minGap=2.5,
                speed_mode=0b00000,
            ),
            num_vehicles=total_vehicles)
        grid_array = {
            "short_length": 100,
            "inner_length": 100,
            "long_length": 100,
            "row_num": 1,
            "col_num": 1,
            "cars_left": int(total_vehicles / 4),
            "cars_right": int(total_vehicles / 4),
            "cars_top": int(total_vehicles / 4),
            "cars_bot": int(total_vehicles / 4)
        }

        additional_net_params = {
            "speed_limit": 35,
            "grid_array": grid_array,
            "horizontal_lanes": 1,
            "vertical_lanes": 1
        }

        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

        self.env, self.scenario = grid_mxn_exp_setup(
            row_num=1,
            col_num=1,
            sumo_params=sumo_params,
            vehicles=vehicles,
            net_params=net_params)

        # go through the env and set all the lights to green
        for i in range(self.env.rows * self.env.cols):
            self.env.traci_connection.trafficlight.setRedYellowGreenState(
                'center' + str(i), "gggggggggggg")

        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, self.scenario)

        self.exp.run(50, 50)

    def test_collide_inflows(self):
        """Tests collisions in the presence of inflows."""
        # create the environment and scenario classes for a ring road
        sumo_params = SumoParams(sim_step=1, render=False)
        total_vehicles = 12
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(SumoCarFollowingController, {}),
            routing_controller=(GridRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                tau=0.1, carFollowModel="Krauss", minGap=2.5,
                speed_mode=0b00000,
            ),
            num_vehicles=total_vehicles)
        grid_array = {
            "short_length": 100,
            "inner_length": 100,
            "long_length": 100,
            "row_num": 1,
            "col_num": 1,
            "cars_left": 3,
            "cars_right": 3,
            "cars_top": 3,
            "cars_bot": 3
        }

        additional_net_params = {
            "speed_limit": 35,
            "grid_array": grid_array,
            "horizontal_lanes": 1,
            "vertical_lanes": 1
        }

        inflows = InFlows()
        inflows.add(veh_type="idm", edge="bot0_0", vehs_per_hour=1000)
        inflows.add(veh_type="idm", edge="top0_1", vehs_per_hour=1000)

        net_params = NetParams(
            no_internal_links=False,
            inflows=inflows,
            additional_params=additional_net_params)

        self.env, self.scenario = grid_mxn_exp_setup(
            row_num=1,
            col_num=1,
            sumo_params=sumo_params,
            vehicles=vehicles,
            net_params=net_params)

        # go through the env and set all the lights to green
        for i in range(self.env.rows * self.env.cols):
            self.env.traci_connection.trafficlight.setRedYellowGreenState(
                'center' + str(i), "gggggggggggg")

        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, self.scenario)

        self.exp.run(50, 50)


if __name__ == '__main__':
    unittest.main()
