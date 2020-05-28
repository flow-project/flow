from run import *
from examples.train import *

def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    dict_args
        dictionary version of the argparse
    """

    # train.py args

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')


    parser.add_argument(
        'exp_title', type=str,
        help='Title to give the run.')


    # optional input parameters
    parser.add_argument(
        '--rl_trainer', type=str, default="rllib",
        help='the RL trainer to use. either rllib or Stable-Baselines')
    parser.add_argument(
        '--load_weights_path', type=str, default=None,
        help='Path to h5 file containing a pretrained model. Relevent for PPO with RLLib'
    )
    parser.add_argument(
        '--algorithm', type=str, default="PPO",
        help='RL algorithm to use. Options are PPO, TD3, MATD3 (MADDPG w/ TD3) right now.'
    )
    parser.add_argument(
        '--num_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over. Relevant for stable-baselines')
    parser.add_argument(
        '--grid_search', action='store_true', default=False,
        help='Whether to grid search over hyperparams')
    parser.add_argument(
        '--num_iterations', type=int, default=200,
        help='How many iterations are in a training run.')
    parser.add_argument(
        '--checkpoint_freq', type=int, default=20,
        help='How often to checkpoint.')
    parser.add_argument(
        '--num_rollouts', type=int, default=1,
        help='How many rollouts are in a training batch')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--local_mode', action='store_true', default=False,
                        help='If true only 1 CPU will be used')
    parser.add_argument('--render', action='store_true', default=False,
                        help='If true, we render the display')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    # Imitation Learning args

    parser.add_argument('--ep_len', type=int, default=5000, help="Maximum length of episode for imitation learning")

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000, help="Number of gradient steps to take per iteration")  # number of gradient steps for training policy
    parser.add_argument('--n_iter', type=int, default=5, help="Number of iterations of DAgger to perform (1st iteration is behavioral cloning)")

    parser.add_argument('--batch_size', type=int, default=3000, help="")  # training data collected (in the env) during each iteration
    parser.add_argument('--init_batch_size', type=int, default=4000)

    parser.add_argument('--train_batch_size', type=int, default=100, help="Batch size for training")  # number of sampled data points to be used per gradient/train step
    parser.add_argument('--tensorboard_path', type=str, help='Path to tensorboard log dir for imitation')

    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--num_eval_episodes', type=int, default=0, help="Number of episodes to evaluate imitation controller.")
    parser.add_argument('--stochastic', type=bool, default=False, help="If true, controller learns stochastic policy (multivariate gaussian)")
    parser.add_argument('--multiagent', type=bool, default=False, help="Whether the env is multiagent")
    parser.add_argument('--v_des', type=float, default=15, help="v_des for FollowerStopper")
    parser.add_argument('--variance_regularizer', type=float, default=0.5, help="Regularization parameter to penalize high variance in negative log likelihood loss")


    parsed_args = parser.parse_known_args(args)[0]
    dict_args = vars(parsed_args)
    dict_args['save_model'] = 1
    dict_args['save_path'] = dict_args['load_weights_path']

    return parsed_args, dict_args



def main(args):

    # Parse args, train imitation learning

    flags, params = parse_args(args)
    params["fcnet_hiddens"] = [32, 32, 32]

    assert flags.n_iter>1, ('DAgger needs >1 iteration')

    print("\n\n********** IMITATION LEARNING ************ \n")
    # run training
    imitation_runner = Runner(params)
    imitation_runner.run_training_loop()

    # convert model to work for PPO and save for training
    imitation_runner.save_controller_for_PPO()

    # Imitation Done, start RL
    print("\n\n********** RL ************ \n")

    # Import relevant information from the exp_config script.
    module = __import__(
        "examples.exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "examples.exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
        multiagent = False
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
        assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
            "Currently, multiagent experiments are only supported through "\
            "RLlib. Try running this experiment using RLlib: " \
            "'python train.py EXP_CONFIG'"
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    if flags.rl_trainer.lower() == "rllib":
        train_rllib(submodule, flags)
    elif flags.rl_trainer.lower() == "stable-baselines":
        train_stable_baselines(submodule, flags)
    elif flags.rl_trainer.lower() == "h-baselines":
        flow_params = submodule.flow_params
        train_h_baselines(flow_params, args, multiagent)
    else:
        raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
                         "or 'stable-baselines'.")

if __name__ == "__main__":
    main(sys.argv[1:])
