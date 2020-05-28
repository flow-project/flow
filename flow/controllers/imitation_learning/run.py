import os
import time
import numpy as np
from trainer import Trainer
from flow.controllers.car_following_models import IDMController


class Runner(object):
    """ Class to run imitation learning (training and evaluation) """

    def __init__(self, params):
        """
        Parameters
        __________
        params: dict
            dictionary of parameters relevent to running imitation learning.
        """

        # initialize trainer class instance and params
        self.params = params

        # import appropriate exp_config module
        if self.params['multiagent']:
            module = __import__("examples.exp_configs.rl.multiagent", fromlist=[self.params['exp_config']])
        else:
            module = __import__("examples.exp_configs.rl.singleagent", fromlist=[self.params['exp_config']])

        submodule = getattr(module, self.params['exp_config'])
        self.trainer = Trainer(params, submodule)

    def run_training_loop(self):
        """
        Runs training for imitation learning for number of iterations specified in params.
        """
        self.trainer.run_training_loop(n_iter=self.params['n_iter'])

    def evaluate(self):
        """
        Evaluates a trained controller over a specified number trajectories; compares average action per step and average reward per trajectory between imitator and expert
        """
        self.trainer.evaluate_controller(num_trajs=self.params['num_eval_episodes'])

    def save_controller_network(self):
        """
        Saves the tensorflow keras model of the imitation policy to a h5 file, whose path is specified in params
        """
        self.trainer.save_controller_network()

    def save_controller_for_PPO(self):
        """
        Creates and saves (in h5 file format) new tensorflow keras model to run PPO with weighs loaded from imitation learning
        """
        self.trainer.save_controller_for_PPO()


def main():
    """
    Parse args, run training, and evaluate.
    """

    import argparse
    parser = argparse.ArgumentParser()

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    parser.add_argument('--ep_len', type=int, default=5000, help='Max length of episodes for rollouts. ')

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000, help='Number of gradient steps for training policy.')  # number of gradient steps for training policy
    parser.add_argument('--n_iter', type=int, default=5, help='Number of DAgger iterations to run (1st iteration is behavioral cloning')

    parser.add_argument('--batch_size', type=int, default=3000, help='Number of environment steps to collect in iteration of DAgger')
    parser.add_argument('--init_batch_size', type=int, default=4000, help='Number of environment steps to collect on 1st iteration of DAgger (behavioral cloning iteration)')

    parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size to train on')

    parser.add_argument('--replay_buffer_size', type=int, default=1000000, help='Max size of replay buffer')
    parser.add_argument('--save_path', type=str, default='', help='Filepath to h5 file in which imitation model should be saved')
    parser.add_argument('--PPO_save_path', type=str, default='', help='Filepath to h5 file in which PPO model with copied weights should be saved')
    parser.add_argument('--save_model', type=int, default=0, help='If true, save models in h5 format')
    parser.add_argument('--num_eval_episodes', type=int, default=0, help='Number of episodes on which to evaluate imitation model')
    parser.add_argument('--stochastic', type=bool, default=False, help='If true, learn a stochastic policy (MV Gaussian)')
    parser.add_argument('--multiagent', type=bool, default=False, help='If true, env is multiagent. ')
    parser.add_argument('--v_des', type=float, default=15, help='Desired velocity for follower-stopper')
    parser.add_argument('--variance_regularizer', type=float, default=0.5, help='Regularization hyperparameter to penalize variance in imitation learning loss, for stochastic policies.')
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    # change this to determine number and size of hidden layers
    params["fcnet_hiddens"] = [32, 32, 32]

    assert args.n_iter>1, ('DAgger needs >1 iteration')


    # run training
    train = Runner(params)
    train.run_training_loop()

    # save model after training
    if params['save_model'] == 1:
        train.save_controller_network()

    # evaluate controller on difference, compared to expert, in action taken and average reward accumulated per rollout
    train.evaluate()

if __name__ == "__main__":
    main()
