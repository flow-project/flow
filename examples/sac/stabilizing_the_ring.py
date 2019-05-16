"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road, using the softlearning library which
implements a soft actor-critic algorithm.
"""

import json
import numpy as np

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray import tune

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, IDMController, ContinuousRouter

from flow.utils.softlearning import ExperimentRunner
from flow.utils.softlearning import get_variant_spec
from flow.utils.softlearning import generate_experiment_kwargs



EPOCHS = 200
HORIZON = 30
N_CHECKPOINTS = 150
N_CPUS = 2
N_GPUS = 0


# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationPOEnv",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

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
        