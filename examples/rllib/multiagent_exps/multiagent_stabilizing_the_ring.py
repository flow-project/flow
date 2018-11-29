"""Ring road example.

Creates a set of stabilizing the ring experiments to test if
 more agents -> fewer needed batches
"""

import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.vehicles import Vehicles
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# make sure (sample_batch_size * num_workers ~= train_batch_size)
# time horizon of a single rollout
HORIZON = 3000
# Number of rings
NUM_RINGS = 1
# number of rollouts per training iteration
N_ROLLOUTS = 20  # int(20/NUM_RINGS)
# number of parallel workers
N_CPUS = 2  # int(20/NUM_RINGS)

# We place one autonomous vehicle and 21 human-driven vehicles in the network
vehicles = Vehicles()
for i in range(NUM_RINGS):
    vehicles.add(
        veh_id='human_{}'.format(i),
        acceleration_controller=(IDMController, {
            'noise': 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)
    vehicles.add(
        veh_id='rl_{}'.format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='lord_of_numrings{}'.format(NUM_RINGS),

    # name of the flow environment the experiment is running on
    env_name='MultiWaveAttenuationPOEnv',

    # name of the scenario class the experiment is running on
    scenario='MultiLoopScenario',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params={
            'max_accel': 1,
            'max_decel': 1,
            'ring_length': [230, 230],
            'target_velocity': 4
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            'length': 230,
            'lanes': 1,
            'speed_limit': 30,
            'resolution': 40,
            'num_rings': NUM_RINGS
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(bunching=20.0, spacing='custom'),
)


def setup_exps():
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5])
    config['horizon'] = HORIZON
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


if __name__ == '__main__':
    alg_run, env_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 1,
            'stop': {
                'training_iteration': 1
            },
            'config': config,
            # 'upload_dir': 's3://<BUCKET NAME>'
        },
    })
