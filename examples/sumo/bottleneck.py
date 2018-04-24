"""
(description)
"""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.scenarios.bridge_toll.gen import BBTollGenerator
from flow.scenarios.bridge_toll.scenario import BBTollScenario
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import ContinuousRouter

from flow.envs.bottleneck_env import BridgeTollEnv
from flow.core.experiment import SumoExperiment

SCALING = 1
DISABLE_TB = True
DISABLE_RAMP_METER = True
FLOW_RATE = 2000 * SCALING  # inflow rate


def bottleneck_example(sumo_binary=None):
    sumo_params = SumoParams(sim_step=0.5,
                             sumo_binary="sumo-gui")

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    vehicles = Vehicles()

    vehicles.add(veh_id="human",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=1621,
                 num_vehicles=1*SCALING)

    additional_env_params = {"target_velocity": 40,
                             "disable_tb": True,
                             "disable_ramp_metering": False}
    env_params = EnvParams(additional_params=additional_env_params,
                           sims_per_step=2,
                           lane_change_duration=1)

    inflow = InFlows()
    inflow.add(veh_type="human", edge="1", vehsPerHour=FLOW_RATE,
               departLane="random", departSpeed=10)

    traffic_lights = TrafficLights()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": SCALING}
    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False,
                           additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="random", min_gap=5,
                                   lanes_distribution=float("inf"),
                                   edges_distribution=["2", "3", "4", "5"])

    scenario = BBTollScenario(name="bay_bridge_toll",
                              generator_class=BBTollGenerator,
                              vehicles=vehicles,
                              net_params=net_params,
                              initial_config=initial_config,
                              traffic_lights=traffic_lights)

    env = BridgeTollEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bottleneck_example()

    # run for a set number of rollouts / time steps
    exp.run(10, 2000)
