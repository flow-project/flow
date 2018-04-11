"""
Example of a multi-lane network with human-driven vehicles.
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights


from flow.scenarios.two_lane_merge_straight.gen import TwoLaneStraightMergeGenerator
from flow.scenarios.two_lane_merge_straight.scenario import TwoLaneMergeScenario
from flow.controllers.lane_change_controllers import *
from flow.controllers.velocity_controllers import FollowerStopper
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.envs.loop_accel import AccelEnv
from flow.core.experiment import SumoExperiment

import numpy as np

def twolanemergestraight(sumo_binary=None):

    if sumo_binary is None:
        sumo_binary = "sumo-gui"
    sumo_params = SumoParams(sim_step = 0.1, sumo_binary="sumo-gui", overtake_right=False)

    vehicles = Vehicles()

    vehicles.add(veh_id="lane_1_human",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 acceleration_controller=(SumoCarFollowingController, {}),
                 # routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0b100000101,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=2.5, tau=1.0, speedDev=0.05),
                 sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0, lcCooperative=0.1),
                 num_vehicles=0)

    vehicles.add(veh_id="lane_0_human",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0b100000101,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=2.5, tau=1.0, speedDev=0.0),
                 sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0, lcCooperative=0.1),
                 num_vehicles=0)

    vehicles.add(veh_id="rl",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0b100000101,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=2.5, tau=1.0, speedDev=0.0),
                 sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0, lcCooperative=0.0),
                 num_vehicles=0)

    additional_env_params = {"target_velocity": 40}
    env_params = EnvParams(additional_params=additional_env_params,
                           lane_change_duration=1)

    high_flow_rate = 2100
    low_flow_rate = 1500

    inflow = InFlows()
    inflow.add(veh_type="lane_0_human", edge="1", probability=high_flow_rate/3600 * 0.90,  # vehsPerHour=veh_per_hour *0.8,
               departLane=0, departSpeed=23)
    inflow.add(veh_type="lane_1_human", edge="1", probability=low_flow_rate/3600,  # vehsPerHour=veh_per_hour *0.8,
               departLane=1, departSpeed=23)
    inflow.add(veh_type="rl", edge="1", probability=high_flow_rate/3600 * 0.10,  # vehsPerHour=veh_per_hour *0.8,
               departLane=0, departSpeed=23)

    net_params = NetParams(in_flows=inflow, no_internal_links=False, additional_params={"merge_type": "priority"})

    initial_config = InitialConfig(spacing="random", min_gap=5,
                                   lanes_distribution=float("inf"),
                                   edges_distribution=["1"])

    scenario = TwoLaneMergeScenario(name="bay_bridge_toll",
                              generator_class=TwoLaneStraightMergeGenerator,
                              vehicles=vehicles,
                              net_params=net_params,
                              initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = twolanemergestraight(sumo_binary="sumo-gui")

    # run for a set number of rollouts / time steps
    exp.run(10, 500)