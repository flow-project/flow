"""Evaluates the baseline performance of figureeight without RL control.

Baseline is human acceleration and intersection behavior.
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
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


def figure_eight_baseline(num_runs, render=True):
    """Run script for all figure eight baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    # We place 1 autonomous vehicle and 13 human-driven vehicles in the network
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(IDMController, {"noise": 0.2}),
                 routing_controller=(ContinuousRouter, {}),
                 sumo_car_following_params=SumoCarFollowingParams(
                     speed_mode="no_collide",
                 ),
                 num_vehicles=14)

    sumo_params = SumoParams(
        sim_step=0.1,
        render=render,
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
