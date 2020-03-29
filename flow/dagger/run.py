import os
import time
import numpy as np
import tensorflow as tf
from trainer import Trainer
from flow.controllers.car_following_models import IDMController


class Runner(object):

    def __init__(self, params):

        # initialize trainer
        self.params = params
        self.trainer = Trainer(params)

    def run_training_loop(self):

        self.trainer.run_training_loop(n_iter=self.params['n_iter'])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep_len', type=int, default=3000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--init_batch_size', type=int, default=5000)

    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--num_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # learning rate for supervised learning
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)

    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')


    # run training
    train = Runner(params)
    train.run_training_loop()

if __name__ == "__main__":
    main()
