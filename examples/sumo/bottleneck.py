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
from flow.core.params import SumoCarFollowingParams
from flow.envs.bottleneck_env import BridgeTollEnv
from flow.core.experiment import BottleneckDensityExperiment
from flow.core.experiment import SumoExperiment
import numpy as np
import ray

SCALING = 1
DISABLE_TB = True
DISABLE_RAMP_METER = False



def bottleneck(flow_rate, horizon, sumo_binary=None):

    if sumo_binary is None:
        sumo_binary = "sumo"
    sumo_params = SumoParams(sim_step = 0.5, sumo_binary=sumo_binary,
                             overtake_right=False, restart_instance=True)

    vehicles = Vehicles()

    vehicles.add(veh_id="human",
                 speed_mode=25,
                 lane_change_controller=(SumoLaneChangeController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=1621,
                 num_vehicles=1*SCALING)

    additional_env_params = {"target_velocity": 40,
                             "disable_tb": True,
                             "disable_ramp_metering": DISABLE_RAMP_METER}
    env_params = EnvParams(horizon=horizon,
                           additional_params=additional_env_params,

                           lane_change_duration=1)

    inflow = InFlows()
    inflow.add(veh_type="human", edge="1", vehsPerHour=flow_rate,
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

    return BottleneckDensityExperiment(env, scenario)


@ray.remote
def run_bottleneck(density, num_trials, num_steps):
    print("Running experiment for density: ", density)
    exp = bottleneck(density, num_steps, sumo_binary="sumo")
    outflow, velocity, bottleneckdensity = exp.run(num_trials, num_steps)
    per_step_avg_velocities = exp.per_step_avg_velocities[:1]
    per_step_densities = exp.per_step_densities[:1]
    per_step_outflows = exp.per_step_outflows[:1]

    return outflow, velocity, bottleneckdensity, per_step_avg_velocities, per_step_densities, per_step_outflows

if __name__ == "__main__":
    # import the experiment variable
    densities = list(range(800,2000,100)) # start stop step
    outflows = []
    velocities = []
    bottleneckdensities = []

    per_step_densities = []
    per_step_avg_velocities = []
    per_step_outflows = []


    #
    # bottleneck_outputs = [run_bottleneck(d, 5, 1500) for d in densities]
    # for output in bottleneck_outputs:

    ray.init(num_cpus=8)
    bottleneck_outputs = [run_bottleneck.remote(d, 5, 2000) for d in densities]
    for output in ray.get(bottleneck_outputs):
        outflow, velocity, bottleneckdensity, per_step_vel, per_step_den, per_step_out = output

        outflows.append(outflow)
        velocities.append(velocity)
        bottleneckdensities.append(bottleneckdensity)

        per_step_densities.extend(per_step_den)
        per_step_avg_velocities.extend(per_step_vel)
        per_step_outflows.extend(per_step_out)

    np.savetxt("rets_alinea.csv", np.matrix([densities, outflows, velocities, bottleneckdensities]).T, delimiter=",")
    np.savetxt("vels_alinea.csv", np.matrix(per_step_avg_velocities), delimiter=",")
    np.savetxt("dens_alinea.csv", np.matrix(per_step_densities), delimiter=",")
    np.savetxt("outflow_alinea.csv", np.matrix(per_step_outflows), delimiter=",")

