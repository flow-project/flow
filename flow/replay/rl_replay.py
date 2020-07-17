"""Replay script for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./rl_replay.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import gym
import numpy as np
from collections import defaultdict
import os
import sys

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

from flow.core.experiment import Experiment

EXAMPLE_USAGE = """
example usage:
    python ./rl_replay.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def read_result_dir(result_dir_path, multi_only=False):
    """Read the provided result_dir and get config and flow_params."""
    result_dir = result_dir_path if result_dir_path[-1] != '/' \
        else result_dir_path[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False
        if multi_only:
            raise NotImplementedError

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)
    return result_dir, config, multiagent, flow_params


def set_sim_params(sim_params, render_mode, save_render, gen_emission):
    """Set up sim_params according to render mode."""
    # hack for old pkl files
    # TODO(ev) remove eventually
    setattr(sim_params, 'num_clients', 1)

    # for hacks for old pkl files TODO: remove eventually
    if not hasattr(sim_params, 'use_ballistic'):
        sim_params.use_ballistic = False

    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if gen_emission else None

    # pick your rendering mode
    if render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif render_mode == 'no_render':
        sim_params.render = False
    if save_render:
        if render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True
    return sim_params


def set_env_params(env_params, evaluate, horizon, config=None):
    """Set up env_params according to commandline arguments."""
    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params.restart_instance = False
    if evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if horizon:
        if config:
            config['horizon'] = horizon
        env_params.horizon = horizon


def set_agents(config, result_dir, env_name, run=None, checkpoint_num=None):
    """Determine and create agents that will be used to compute actions."""
    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if run and config_run:
        if run != config_run:
            print('rl_replay.py: error: run argument '
                  + '\'{}\' passed in '.format(run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if run:
        agent_cls = get_agent_class(run)
    elif config['env_config']['run'] == "<class 'ray.rllib.agents.trainer_template.CCPPOTrainer'>":
        from flow.algorithms.centralized_PPO import CCTrainer, CentralizedCriticModel
        from ray.rllib.models import ModelCatalog
        agent_cls = CCTrainer
        ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    elif config['env_config']['run'] == "<class 'ray.rllib.agents.trainer_template.CustomPPOTrainer'>":
        from flow.algorithms.custom_ppo import CustomPPOTrainer
        agent_cls = CustomPPOTrainer
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('rl_replay.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./rl_replay.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
    agent.restore(checkpoint)

    return agent


def get_rl_action(config, agent, multiagent, multi_only=False):
    """Return a function that compute action based on a given state."""
    policy_map_fn = None
    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            size = config['model']['lstm_cell_size']
            state_init = defaultdict(lambda: [np.zeros(size, np.float32),
                                              np.zeros(size, np.float32)])
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    def rl_action(state):
        if multiagent:
            action = {}
            for agent_id in state.keys():
                if use_lstm:
                    action[agent_id], state_init[agent_id], logits = \
                        agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                else:
                    action[agent_id] = agent.compute_action(
                        state[agent_id], policy_id=policy_map_fn(agent_id))
        else:
            if use_lstm and multi_only:
                raise NotImplementedError
            action = agent.compute_action(state)
        return action
    return policy_map_fn, rl_action, rets


def replay_rllib(args):
    """Replay for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    replay script), and renders the experiment associated with it.
    """
    result_dir, config, multiagent, flow_params = read_result_dir(args.result_dir)

    sim_params = set_sim_params(flow_params['sim'], args.render_mode,
                                args.save_render, args.gen_emission)

    # Create and register a gym+rllib env
    exp = Experiment(flow_params, register_with_ray=True)
    register_env(exp.env_name, exp.create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'
    set_env_params(flow_params['env'], args.evaluate, args.horizon, config)

    agent = set_agents(config, result_dir, exp.env_name, run=args.run, checkpoint_num=args.checkpoint_num)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        exp.env = agent.local_evaluator.env
    else:
        exp.env = gym.make(exp.env_name)

    # reroute on exit is a training hack, it should be turned off at test time.
    if hasattr(exp.env, "reroute_on_exit"):
        exp.env.reroute_on_exit = False

    if args.render_mode == 'sumo_gui':
        exp.env.sim_params.render = True  # set to True after initializing agent and env

    rl_action, policy_map_fn, rets = get_rl_action(config, agent, multiagent)

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        exp.env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    exp.run(num_runs=args.num_rollouts, convert_to_csv=args.gen_emission, to_aws=args.to_aws,
            rl_actions=rl_action, multiagent=multiagent, rets=rets, policy_map_fn=policy_map_fn)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to replay.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    parser.add_argument(
        '--is_baseline',
        action='store_true',
        help='specifies whether this is a baseline run'
    )
    parser.add_argument(
        '--to_aws',
        type=str, nargs='?', default=None, const="default",
        help='Specifies the name of the partition to store the output'
             'file on S3. Putting not None value for this argument'
             'automatically set gen_emission to True.'
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    rl_replay(args)
