"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import random
import json

from flow.core.experiment import Experiment

from hbaselines.fcnet.td3 import FeedForwardPolicy \
    as TD3FeedForwardPolicy
from hbaselines.fcnet.sac import FeedForwardPolicy \
    as SACFeedForwardPolicy
from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy \
    as TD3GoalConditionedPolicy
from hbaselines.goal_conditioned.sac import GoalConditionedPolicy \
    as SACGoalConditionedPolicy
from hbaselines.envs.mixed_autonomy.params.highway \
    import get_flow_params as highway
from hbaselines.envs.mixed_autonomy.params.i210 \
    import get_flow_params as i210

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

# dictionary that maps policy names to policy objects
POLICY_DICT = {
    "FeedForwardPolicy": {
        "TD3": TD3FeedForwardPolicy,
        "SAC": SACFeedForwardPolicy,
    },
    "GoalConditionedPolicy": {
        "TD3": TD3GoalConditionedPolicy,
        "SAC": SACGoalConditionedPolicy,
    },
}


ENV_ATTRIBUTES = {
    "highway-v0": lambda multiagent: highway(
        fixed_boundary=True,
        stopping_penalty=True,
        acceleration_penalty=True,
        multiagent=multiagent,
    ),
    "highway-v1": lambda multiagent: highway(
        fixed_boundary=True,
        stopping_penalty=False,
        acceleration_penalty=True,
        multiagent=multiagent,
    ),
    "highway-v2": lambda multiagent: highway(
        fixed_boundary=True,
        stopping_penalty=False,
        acceleration_penalty=False,
        multiagent=multiagent,
    ),
    "i210-v0": lambda multiagent: i210(
        fixed_boundary=True,
        stopping_penalty=True,
        acceleration_penalty=True,
        multiagent=multiagent,
    ),
    "i210-v1": lambda multiagent: i210(
        fixed_boundary=True,
        stopping_penalty=False,
        acceleration_penalty=True,
        multiagent=multiagent,
    ),
    "i210-v2": lambda multiagent: i210(
        fixed_boundary=True,
        stopping_penalty=False,
        acceleration_penalty=False,
        multiagent=multiagent,
    ),
}


def parse_options(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description='Run evaluation episodes of a given checkpoint.',
        epilog='python run_eval "/path/to/dir_name" ckpt_num')

    # required input parameters
    parser.add_argument(
        'dir_name', type=str, help='the path to the checkpoints folder')

    # optional arguments
    parser.add_argument(
        '--ckpt_num', type=int, default=None,
        help='the checkpoint number. If not specified, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1,
        help='number of eval episodes')
    parser.add_argument(
        '--no_render', action='store_true',
        help='shuts off rendering')
    parser.add_argument(
        '--random_seed', action='store_true',
        help='whether to run the simulation on a random seed. If not added, '
             'the original seed is used.')
    parser.add_argument(
        '--to_aws', action='store_true',
        help='shuts off rendering')

    flags, _ = parser.parse_known_args(args)

    return flags


def get_hyperparameters_from_dir(ckpt_path):
    """Collect the algorithm-specific hyperparameters from the checkpoint.

    Parameters
    ----------
    ckpt_path : str
        the path to the checkpoints folder

    Returns
    -------
    str
        environment name
    hbaselines.goal_conditioned.*
        policy object
    dict
        algorithm and policy hyperparaemters
    int
        the seed value
    """
    # import the dictionary of hyperparameters
    with open(os.path.join(ckpt_path, 'hyperparameters.json'), 'r') as f:
        hp = json.load(f)

    # collect the policy object
    policy_name = hp['policy_name']
    alg_name = hp['algorithm']
    policy = POLICY_DICT[policy_name][alg_name]

    # collect the environment name
    env_name = hp['env_name']

    # collect the seed value
    seed = hp['seed']

    # remove unnecessary features from hp dict
    hp = hp.copy()
    del hp['policy_name'], hp['env_name'], hp['seed']
    del hp['algorithm'], hp['date/time']

    return env_name, policy, hp, seed


def get_flow_params(env_name):
    """Read the provided result_dir and get config and flow_params.

    Parameters
    ----------
    env_name : str
        the name of the environment (in h-baselines)

    Returns
    -------
    dict
        the flow-params dict object
    """
    # Handle multi-agent environments.
    multiagent = env_name.startswith("multiagent")
    if multiagent:
        env_name = env_name[11:]

    flow_params = ENV_ATTRIBUTES[env_name](multiagent)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    flow_params["sim"].emission_path = emission_path

    flow_params["sim"].render = True

    return flow_params


def get_rl_actions(policy_tf, ob_space):
    """Define the method through which actions are assigned to the RL agent(s).

    Parameters
    ----------
    policy_tf : hbaselines.base_policies.*
        the policy object
    ob_space : gym.space.*
        the observation space

    Returns
    -------
    method
        the rl_actions method to use in the Experiment object
    """
    def rl_actions(obs):
        """Get the actions from a given observation.

        Parameters
        ----------
        obs : array_like
            the observation

        Returns
        -------
        list of float
            the action value
        """
        # Reshape the observation to match the input structure of the policy.
        if isinstance(obs, dict):
            # In multi-agent environments, observations come in dict form
            for key in obs.keys():
                # Shared policies with have one observation space, while
                # independent policies have a different observation space based
                # on their agent ID.
                if isinstance(ob_space, dict):
                    ob_shape = ob_space[key].shape
                else:
                    ob_shape = ob_space.shape
                obs[key] = np.array(obs[key]).reshape((-1,) + ob_shape)
        else:
            obs = np.array(obs).reshape((-1,) + ob_space.shape)

        action = policy_tf.get_action(
            obs, None,
            apply_noise=False,
            random_actions=False,
            env_num=0,
        )

        # Flatten the actions. Dictionaries correspond to multi-agent policies.
        if isinstance(action, dict):
            action = {key: action[key].flatten() for key in action.keys()}
        else:
            action = action.flatten()

        return action

    return rl_actions


def main(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    # ======================================================================= #
    # Step 1: Import relevant data.                                           #
    # ======================================================================= #

    flags = parse_options(args)

    # get the hyperparameters
    env_name, policy, hp, seed = get_hyperparameters_from_dir(flags.dir_name)
    hp['num_envs'] = 1

    # Get the flow-specific parameters.
    flow_params = get_flow_params(env_name)

    # setup the seed value
    if not flags.random_seed:
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

    # Create the experiment object.
    exp = Experiment(flow_params)

    # ======================================================================= #
    # Step 2: Setup the policy.                                               #
    # ======================================================================= #

    # Create a tensorflow session.
    sess = tf.compat.v1.Session()

    # Get the checkpoint number.
    if flags.ckpt_num is None:
        filenames = os.listdir(os.path.join(flags.dir_name, "checkpoints"))
        metafiles = [f[:-5] for f in filenames if f[-5:] == ".meta"]
        metanum = [int(f.split("-")[-1]) for f in metafiles]
        ckpt_num = max(metanum)
    else:
        ckpt_num = flags.ckpt_num

    # location to the checkpoint
    ckpt = os.path.join(flags.dir_name, "checkpoints/itr-{}".format(ckpt_num))

    # Create the policy.
    policy_tf = policy(
        sess=sess,
        ob_space=exp.env.observation_space,
        ac_space=exp.env.action_space,
        co_space=None,
        verbose=2,
        layers=[256, 256],
        act_fun=tf.nn.relu,
        fingerprint_range=([0, 0], [5, 5]),
        env_name=env_name,
        **hp["policy_kwargs"]
    )

    trainable_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    # Restore the previous checkpoint.
    saver = tf.compat.v1.train.Saver(trainable_vars)
    saver.restore(sess, ckpt)

    # Create a method to compute the RL actions.
    rl_actions = get_rl_actions(policy_tf, exp.env.observation_space)

    # ======================================================================= #
    # Step 3: Setup the and run the experiment.                               #
    # ======================================================================= #

    exp.run(
        num_runs=flags.num_rollouts,
        convert_to_csv=True,
        to_aws=flags.to_aws,
        rl_actions=rl_actions,
        multiagent=env_name.startswith("multiagent"),
    )


if __name__ == '__main__':
    main(sys.argv[1:])
