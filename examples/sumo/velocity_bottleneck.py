"""
Example of a multi-lane network with human-driven vehicles.
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.lane_change_controllers import *
from flow.controllers.velocity_controllers import HandTunedVelocityController, FeedbackController
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import SumoLaneChangeParams
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.core.experiment import SumoExperiment

import numpy as np

def bottleneck(sumo_binary=None):

    SCALING = 1
    NUM_LANES = 4*SCALING  # number of lanes in the widest highway
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    AV_FRAC = .2

    if sumo_binary is None:
        sumo_binary = "sumo-gui"
    sumo_params = SumoParams(sim_step = 0.5, sumo_binary=sumo_binary, overtake_right=False, restart_instance=True)

    vehicles = Vehicles()

    vehicles.add(veh_id="human",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 acceleration_controller=(SumoCarFollowingController, {}),
                 # routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0b100000101,
                 sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0),
                 num_vehicles=5)

    vehicles.add(veh_id="followerstopper",
                 lane_change_controller=(SumoLaneChangeController, {}),
                 # acceleration_controller=(HandTunedVelocityController, {"v_regions":[23, 5, 1, 60, 60, 60, 60, 60, 60]}),
                 acceleration_controller=(FeedbackController, {"K":50, "desired_bottleneck_density":0.0025, "danger_edges":["3", "4", "5"]}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=0b100000101,
                 sumo_lc_params=SumoLaneChangeParams(lcKeepRight=0),
                 speed_mode=9,
                 num_vehicles=5)

    horizon = 500
    num_segments = [("1", 1, False), ("2", 3, True), ("3", 3, True),
                    ("4", 1, True), ("5", 1, False)]
    additional_env_params = {"target_velocity": 40, "num_steps": horizon,
                             "disable_tb": True, "disable_ramp_metering": True,
                             "segments": num_segments}
    env_params = EnvParams(additional_params=additional_env_params,
                           lane_change_duration=1)

    # flow rate

    # MAX OF 3600 vehicles per lane per hour i.e. flow_rate <= 3600 *
    flow_rate = 2000 * SCALING
    # percentage of flow coming out of each lane
    # flow_dist = np.random.dirichlet(np.ones(NUM_LANES), size=1)[0]
    flow_dist = np.ones(NUM_LANES)/NUM_LANES

    inflow = InFlows()
    inflow.add(veh_type="human", edge="1", vehs_per_hour=flow_rate*(1-AV_FRAC),#vehsPerHour=veh_per_hour *0.8,
               departLane="random", departSpeed=10)
    inflow.add(veh_type="followerstopper", edge="1", vehs_per_hour=flow_rate*AV_FRAC,#vehsPerHour=veh_per_hour * 0.2,
               departLane="random", departSpeed=10)

    traffic_lights = TrafficLights()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": SCALING}
    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="random", min_gap=5,
                                   lanes_distribution=float("inf"),
                                   edges_distribution=["2", "3", "4", "5"])

    scenario = BBTollScenario(name="bay_bridge_toll",
                              generator_class=BBTollGenerator,
                              vehicles=vehicles,
                              net_params=net_params,
                              initial_config=initial_config,
                              traffic_lights=traffic_lights)

    env = DesiredVelocityEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bottleneck(sumo_binary="sumo-gui")

    # run for a set number of rollouts / time steps
    exp.run(5, 500)
    print(exp.rollout_total_rewards)
    # print(exp.per_step_rewards[0])
    # np.savetxt("rets.csv", np.array(exp.per_step_rewards), delimiter=",")