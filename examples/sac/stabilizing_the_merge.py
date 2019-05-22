"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""
import numpy as np

import ray
from ray import tune

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController

from flow.utils.softlearning import ExperimentRunner
from flow.utils.softlearning import get_variant_spec
from flow.utils.softlearning import generate_experiment_kwargs

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0


EPOCHS = 400
HORIZON = 2000
N_CHECKPOINTS = 40
N_CPUS = 1
N_GPUS = 0


# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="exp_stab_merge_sac",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationMergePOEnv",

    # name of the scenario class the experiment is running on
    scenario="MergeScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


sac_params = dict(
    flow_params=flow_params,

    algorithm_params={
        'type': 'SAC',

        'kwargs': {
            'n_epochs': EPOCHS,
            'epoch_length': HORIZON,
            'train_every_n_steps': 1,
            'n_train_repeat': 1,
            'eval_render_mode': None,
            'eval_n_episodes': 1,
            'eval_deterministic': True,
            'discount': 0.99,
            'tau': 5e-3,
            'reward_scale': 1.0,
            'reparameterize': True,
            'lr': 3e-4,
            'target_update_interval': 1,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },

    gaussian_policy_params={
        'type': 'GaussianPolicy',
        'kwargs': {
            'hidden_layer_sizes': (256, 256),
            'squash': True,
        }
    },

    sampler_params={
        'type': 'SimpleSampler',
        'kwargs': {
            'max_path_length': HORIZON,
            'min_pool_size': HORIZON,
            'batch_size': 256,
        }
    },

    run_params={
        'seed': tune.sample_from(
            lambda spec: np.random.randint(0, 10000)),
        'checkpoint_at_end': True,
        'checkpoint_frequency': HORIZON // N_CHECKPOINTS,
        'checkpoint_replay_pool': False,
    },

    resources_per_trial={
        'cpu': N_CPUS,
        'gpu': N_GPUS,
        'extra_cpu': 0,
        'extra_gpu': 0,
    }
)


if __name__ == "__main__":
    trainable_class = ExperimentRunner
    variant_spec = get_variant_spec(sac_params)
    experiment_kwargs = generate_experiment_kwargs(variant_spec)

    ray.init(
        num_cpus=N_CPUS,
        num_gpus=N_GPUS)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        reuse_actors=True)
