import os
import time
import numpy as np
from trainer import Trainer
from flow.controllers.car_following_models import IDMController


class Runner(object):
    """ Class to run imitation learning (training and evaluation) """

    def __init__(self, params):

        # initialize trainer class instance and params
        self.params = params
        self.trainer = Trainer(params)

    def run_training_loop(self):
        """
        Runs training for imitation learning for specified number of iterations
        """
        self.trainer.run_training_loop(n_iter=self.params['n_iter'])

    def evaluate(self):
        """
        Evaluates a trained controller over a specified number trajectories; compares average action per step and average reward per trajectory between imitator and expert
        """
        self.trainer.evaluate_controller(num_trajs=self.params['num_eval_episodes'])

    def save_controller_network(self):
        """
        Saves a tensorflow checkpoint to path specified in params (and writes to tensorboard)
        """
        self.trainer.save_controller_network()


def main():
    """
    Parse args, run training, and evalutation
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep_len', type=int, default=5000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=3000)  # training data collected (in the env) during each iteration
    parser.add_argument('--init_batch_size', type=int, default=4000)

    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--num_layers', type=int, default=3)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # learning rate for supervised learning
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--num_eval_episodes', type=int, default=30)
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--noise_variance',type=float, default=0.5)
    parser.add_argument('--vehicle_id', type=str, default='rl_0')
    parser.add_argument('--multiagent', type=bool, default=False)
    parser.add_argument('--v_des', type=float, default=15)

    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)
    assert args.n_iter>1, ('DAgger needs >1 iteration')


    # run training
    train = Runner(params)
    train.run_training_loop()

    # save model after training
    if params['save_model'] == 1:
        train.save_controller_network()


    # evaluate
    train.evaluate()
    print("DONE")


if __name__ == "__main__":
    main()
