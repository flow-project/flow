"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road, using the softlearning library which
implements a soft actor-critic algorithm.
"""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, IDMController, ContinuousRouter

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 2

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


from softlearning.misc.utils import deep_update
from ray import tune
import numpy as np

M = 256
REPARAMETERIZE = True

DEFAULT_MAX_PATH_LENGTH = 3000
DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 20

GAUSSIAN_POLICY_PARAMS = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

ALGORITHM_PARAMS = {
    'type': 'SAC',

    'kwargs': {
        'n_epochs': DEFAULT_NUM_EPOCHS,
        'epoch_length': HORIZON,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'reparameterize': REPARAMETERIZE,
        'lr': 3e-4,
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'store_extra_policy_info': False,
        'action_prior': 'uniform',
        'n_initial_exploration_steps': int(1e3),
    }
}

def get_variant_spec_base():
    variant_spec = {
        # 'git_sha': get_git_rev(__file__),
        'env_config': {
            'flow_params': json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
        },
        'flow_params': flow_params,        
        'environment_params': {
            'training': {
                'kwargs': {
                    # Add environment params here
                },
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': GAUSSIAN_POLICY_PARAMS,
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': ALGORITHM_PARAMS,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': DEFAULT_MAX_PATH_LENGTH,
                'min_pool_size': DEFAULT_MAX_PATH_LENGTH,
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': DEFAULT_NUM_EPOCHS // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec

variant_spec = get_variant_spec_base()

from exprunner import ExperimentRunner

import multiprocessing
from softlearning.misc.utils import datetimestamp

def generate_experiment_kwargs(variant_spec):
    local_dir = '~/ray_results/SAC_ring/'

    resources_per_trial = {}
    resources_per_trial['cpu'] = N_CPUS#multiprocessing.cpu_count()
    resources_per_trial['gpu'] = 0#None
    resources_per_trial['extra_cpu'] = 0#None
    resources_per_trial['extra_gpu'] = 0#None

    datetime_prefix = datetimestamp()
    experiment_id = '-'.join((datetime_prefix, 'experiment_name'))

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None

        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)

        return tune.function(trial_name_creator)

    experiment_kwargs = {
        'name': experiment_id,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': 1,
        'upload_dir': '',
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'trial_name_creator': create_trial_name_creator(
            'id={trial.trial_id}-seed={trial.config[run_params][seed]}'),
        'restore': None,
    }

    return experiment_kwargs

if __name__ == "__main__":
    # alg_run, gym_name, config, env = setup_exps()

    trainable_class = ExperimentRunner

    experiment_kwargs = generate_experiment_kwargs(variant_spec)

    print("\n\nEXPERIMENT_KWARGS", experiment_kwargs, "\n\n\n")

    ray.init(
        num_cpus=N_CPUS, num_gpus=0, resources={}, local_mode=False,
        )#include_webui=True, temp_dir='~/tmp_tmp')#,
        # resources=example_args.resources or {},
        # local_mode=local_mode,
        # include_webui=example_args.include_webui, TODO
        # temp_dir=example_args.temp_dir)

    tune.run(
        trainable_class,
        **experiment_kwargs,
        with_server=False,
        server_port=9898,
        scheduler=None,
        reuse_actors=True)