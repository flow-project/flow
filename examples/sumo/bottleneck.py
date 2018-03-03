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
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.envs.bottleneck_env import BottleNeckEnv
from flow.core.experiment import SumoExperiment


def bottleneck(sumo_binary=None):

    SCALING = 4
    NUM_LANES = 4*SCALING  # number of lanes in the widest highway
    DISABLE_TB = True
    DISABLE_RAMP_METER = True

    logging.basicConfig(level=logging.INFO)

    if sumo_binary is None:
        sumo_binary = "sumo-gui"
    sumo_params = SumoParams(sim_step = 0.5, sumo_binary="sumo-gui")

    vehicles = Vehicles()

    vehicles.add(veh_id="human",
                 speed_mode=31,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 # acceleration_controller=(IDMController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=1621,
                 sumo_car_following_params=SumoCarFollowingParams(
                     minGap=2.5, tau=1.0, speedDev=0.5),
                 num_vehicles=40*SCALING)

    additional_env_params = {"target_velocity": 40, "disable_tb": True,
                             "disable_ramp_metering": True}
    env_params = EnvParams(additional_params=additional_env_params,
                           lane_change_duration=1)

    # flow rate
    flow_rate = 1500 * SCALING
    # percentage of flow coming out of each lane
    #flow_dist = np.random.dirichlet(np.ones(NUM_LANES), size=1)[0]
    flow_dist = np.ones(NUM_LANES)/NUM_LANES

    inflow = InFlows()
    for i in range(NUM_LANES):
        lane_num = str(i)
        veh_per_hour = flow_rate * flow_dist[i]
        inflow.add(veh_type="human", edge="1", vehsPerHour=veh_per_hour,
                   departLane=lane_num, departSpeed=10)

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

    env = BottleNeckEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bottleneck(sumo_binary="sumo-gui")

    # run for a set number of rollouts / time steps
    exp.run(10, 300)