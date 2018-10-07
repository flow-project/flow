"""Evaluates the baseline performance of merge without RL control.

Baseline is no AVs.
"""

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.scenarios.merge.scenario import ADDITIONAL_NET_PARAMS
from flow.core.vehicles import Vehicles
from flow.core.experiment import SumoExperiment
from flow.controllers import SumoCarFollowingController
from flow.scenarios.merge.scenario import MergeScenario
from flow.scenarios.merge.gen import MergeGenerator
from flow.envs.merge import WaveAttenuationMergePOEnv
import numpy as np

# time horizon of a single rollout
HORIZON = 750
# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = 0.1
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = 5


def merge_baseline(num_runs, render=True):
    """Run script for all merge baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render: bool, optional
            specifies whether to use sumo's gui during execution

    Returns
    -------
        SumoExperiment
            class needed to run simulations
    """
    # We consider a highway network with an upstream merging lane producing
    # shockwaves
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 500

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = Vehicles()
    vehicles.add(veh_id="human",
                 acceleration_controller=(SumoCarFollowingController, {}),
                 sumo_car_following_params=SumoCarFollowingParams(
                     speed_mode=9,
                 ),
                 num_vehicles=5)

    # Vehicles are introduced from both sides of merge, with RL vehicles
    # entering from the highway portion as well
    inflow = InFlows()
    inflow.add(veh_type="human", edge="inflow_highway",
               vehs_per_hour=FLOW_RATE,
               departLane="free", departSpeed=10)
    inflow.add(veh_type="human", edge="inflow_merge", vehs_per_hour=100,
               departLane="free", departSpeed=7.5)

    sumo_params = SumoParams(
        restart_instance=True,
        sim_step=0.5,  # time step decreased to prevent occasional crashes
        render=render,
    )

    env_params = EnvParams(
        horizon=HORIZON,
        sims_per_step=5,  # value raised to ensure sec/step match experiment
        warmup_steps=0,
        evaluate=True,  # Set to True to evaluate traffic metric performance
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    )

    initial_config = InitialConfig()

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
    )

    scenario = MergeScenario(name="merge",
                             generator_class=MergeGenerator,
                             vehicles=vehicles,
                             net_params=net_params,
                             initial_config=initial_config)

    env = WaveAttenuationMergePOEnv(env_params, sumo_params, scenario)

    exp = SumoExperiment(env, scenario)

    results = exp.run(num_runs, HORIZON)
    avg_speed = np.mean(results["mean_returns"])

    return avg_speed


if __name__ == "__main__":
    runs = 2  # number of simulations to average over
    res = merge_baseline(num_runs=runs, render=False)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
