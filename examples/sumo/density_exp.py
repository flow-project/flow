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
from flow.controllers.car_following_models import SumoCarFollowingController
from flow.core.params import SumoLaneChangeParams
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.core.experiment import BottleneckDensityExperiment

import ray

import numpy as np

def bottleneck(flow_rate, horizon, sumo_binary=None):

    SCALING = 1
    NUM_LANES = 4*SCALING  # number of lanes in the widest highway
    DISABLE_TB = True
    DISABLE_RAMP_METER = True

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

    num_segments = [("1", 1, False), ("2", 3, False), ("3", 3, False), ("4", 1, False), ("5", 1, False)]
    additional_env_params = {"target_velocity": 40, "num_steps": horizon,
                             "disable_tb": True, "disable_ramp_metering": True,
                             "segments": num_segments}
    env_params = EnvParams(additional_params=additional_env_params,
                           lane_change_duration=1)

    # flow_dist = np.ones(NUM_LANES)/NUM_LANES

    inflow = InFlows()
    inflow.add(veh_type="human", edge="1", vehs_per_hour=flow_rate,#vehsPerHour=veh_per_hour *0.8,
               departLane="random", departSpeed=10)
    # for i in range(NUM_LANES):
    #     lane_num = str(i)
    #     veh_per_hour = flow_rate * flow_dist[i]
    #     veh_per_second = veh_per_hour/3600
    #     inflow.add(veh_type="human", edge="1", probability=veh_per_second,
    #                departLane=lane_num, departSpeed=10)

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

    return BottleneckDensityExperiment(env, scenario)

# @ray.remote
def run_bottleneck(density, num_trials, num_steps):
    exp = bottleneck(density, num_steps, sumo_binary="sumo-gui")
    outflow, velocity, bottleneckdensity = exp.run(num_trials, num_steps)
    per_step_avg_velocities = exp.per_step_avg_velocities[:1]
    per_step_densities = exp.per_step_densities[:1]
    per_step_rewards = exp.per_step_rewards[:1]

    return outflow, velocity, bottleneckdensity, per_step_avg_velocities, per_step_densities, per_step_rewards

if __name__ == "__main__":
    # import the experiment variable
    densities = list(range(800,3001,100))
    outflows = []
    velocities = []
    bottleneckdensities = []

    per_step_densities = []
    per_step_avg_velocities = []
    per_step_rewards = []



    bottleneck_outputs = [run_bottleneck(d, 1, 100) for d in densities]
    for output in bottleneck_outputs:

    # ray.init()
    # bottleneck_outputs = [run_bottleneck.remote(d, 30, 2500) for d in densities]
    # for output in ray.get(bottleneck_outputs):
        outflow, velocity, bottleneckdensity, per_step_vel, per_step_den, per_step_r = output

        outflows.append(outflow)
        velocities.append(velocity)
        bottleneckdensities.append(bottleneckdensity)

        per_step_densities.extend(per_step_den)
        per_step_avg_velocities.extend(per_step_vel)
        per_step_rewards.extend(per_step_r)

    np.savetxt("rets.csv", np.matrix([densities, outflows, velocities, bottleneckdensities]).T, delimiter=",")
    np.savetxt("vels.csv", np.matrix(per_step_avg_velocities), delimiter=",")
    np.savetxt("dens.csv", np.matrix(per_step_densities), delimiter=",")
    np.savetxt("outflow.csv", np.matrix(per_step_rewards), delimiter=",")

