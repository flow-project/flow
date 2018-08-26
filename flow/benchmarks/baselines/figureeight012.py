"""
Script to evaluate the baseline performance of figureeight without RL control
Baseline is human intersection behavior

Trains a fraction of vehicles in a ring road structure to regulate the flow of
vehicles through an intersection. In this example, the last vehicle in the
network is an autonomous vehicle.

Action Dimension: (1, )

Observation Dimension: (28, )

Horizon: 1500 steps
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS
from flow.envs.loop.loop_accel import AccelEnv
from flow.core.experiment import SumoExperiment
import numpy as np

# time horizon of a single rollout
HORIZON = 1500


def figure_eight_baseline(num_runs, sumo_binary="sumo-gui"):
    # We place 1 autonomous vehicle and 13 human-driven vehicles in the network
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(IDMController, {"noise": 0.2}),
                 routing_controller=(ContinuousRouter, {}),
                 speed_mode="no_collide",
                 num_vehicles=14)

    sumo_params = SumoParams(
        sim_step=0.1,
        sumo_binary=sumo_binary,
    )

    env_params = EnvParams(
        horizon=HORIZON,
        evaluate=True,  # Set to True to evaluate traffic metrics
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
        },
    )

    initial_config = InitialConfig()

    net_params = NetParams(
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS,
    )

    scenario = Figure8Scenario(name="figure_eight",
                               generator_class=Figure8Generator,
                               vehicles=vehicles,
                               net_params=net_params,
                               initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, HORIZON)
    avg_speed = np.mean(results["mean_returns"])

    return avg_speed


if __name__ == "__main__":
    runs = 2  # number of simulations to average over
    res = figure_eight_baseline(num_runs=runs)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
