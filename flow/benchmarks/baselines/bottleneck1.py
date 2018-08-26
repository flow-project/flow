"""
This script is used to quickly evaluate a baseline for bottleneck1.
Baseline is no AVs.

Bottleneck in which the actions are specifying a desired velocity in a segment
of space. The autonomous penetration rate in this example is 25%.
Humans are allowed to lane change

Action Dimension: (?, )

Observation Dimension: (?, )

Horizon: 1000 steps
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles
from flow.controllers import ContinuousRouter
from flow.envs.bottleneck_env import DesiredVelocityEnv
from flow.core.experiment import SumoExperiment
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.scenarios.bottleneck.gen import BottleneckGenerator
import numpy as np

# time horizon of a single rollout
HORIZON = 1000

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.25


def bottleneck1_baseline(num_runs, sumo_binary="sumo-gui"):
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 speed_mode=9,
                 routing_controller=(ContinuousRouter, {}),
                 lane_change_mode=1621,
                 num_vehicles=1 * SCALING)

    controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                           ("4", 2, True), ("5", 1, False)]
    num_observed_segments = [("1", 1), ("2", 3), ("3", 3),
                             ("4", 3), ("5", 1)]
    additional_env_params = {
        "target_velocity": 40,
        "disable_tb": True,
        "disable_ramp_metering": True,
        "controlled_segments": controlled_segments,
        "symmetric": False,
        "observed_segments": num_observed_segments,
        "reset_inflow": False,
        "lane_change_duration": 5,
        "max_accel": 3,
        "max_decel": 3,
        "inflow_range": [1000, 2000]
    }

    # flow rate
    flow_rate = 1900 * SCALING

    # percentage of flow coming out of each lane
    inflow = InFlows()
    inflow.add(veh_type="human", edge="1",
               vehs_per_hour=flow_rate,
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

    sumo_params = SumoParams(
        sim_step=0.5,
        sumo_binary=sumo_binary,
        print_warnings=False,
        restart_instance=False,
    )

    env_params = EnvParams(
        evaluate=True,  # Set to True to evaluate traffic metrics
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    )

    initial_config = InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    )

    scenario = BottleneckScenario(name="bay_bridge_toll",
                                  generator_class=BottleneckGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config,
                                  traffic_lights=traffic_lights)

    env = DesiredVelocityEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, HORIZON)
    avg_outflow = np.mean([outflow[-1]
                           for outflow in results["per_step_returns"]])

    return avg_outflow


if __name__ == "__main__":
    runs = 2  # number of simulations to average over
    res = bottleneck1_baseline(num_runs=runs)

    print('---------')
    print('The average outflow over 500 seconds '
          'across {} runs is {}'.format(runs, res))
