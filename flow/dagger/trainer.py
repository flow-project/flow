import time
from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os
from flow.utils.registry import make_create_env
from env_params_test import flow_params
from imitating_controller2 import ImitatingController
from flow.controllers.car_following_models import IDMController
from flow.core.params import SumoCarFollowingParams
from utils import *

class Trainer(object):

    def __init__(self, params):
        self.params = params
        self.sess = create_tf_session()

        # TODO: replace this with appropriate Flow env
        # print('ERROR CHECK ', flow_params_test['exp_tag'])
        create_env, _ = make_create_env(flow_params)
        self.env = create_env()
        self.env.reset()

        assert 'rl_0' in self.env.k.vehicle.get_ids()
        self.vehicle_id = 'rl_0'

        obs_dim = self.env.observation_space.shape[0]

        # TODO: make sure this is correct
        action_dim = (1,)[0]
        self.params['action_dim'] = action_dim
        self.params['obs_dim'] = obs_dim

        car_following_params = SumoCarFollowingParams()
        self.controller = ImitatingController(self.vehicle_id, self.sess, self.params['action_dim'], self.params['obs_dim'], self.params['num_layers'], self.params['size'], self.params['learning_rate'], self.params['replay_buffer_size'], car_following_params = car_following_params)
        self.expert_controller = IDMController(self.vehicle_id, car_following_params = car_following_params)

        tf.global_variables_initializer().run(session=self.sess)


    def run_training_loop(self, n_iter):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            if itr == 0:
                training_returns = self.collect_training_trajectories(itr, self.params['init_batch_size'])
            else:
                training_returns = self.collect_training_trajectories(itr, self.params['batch_size'])

            paths, envsteps_this_batch = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.controller.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            loss = self.train_controller()

    def collect_training_trajectories(self, itr, batch_size):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        if itr == 0:
            collect_controller = self.expert_controller
        else:
            collect_controller = self.controller

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, self.vehicle_id, collect_controller, self.expert_controller, batch_size, self.params['ep_len'])

        return paths, envsteps_this_batch

    def train_controller(self):
        print('Training controller using sampled data from replay buffer')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # TODO: fix this
            ob_batch, ac_batch, expert_ac_batch, re_batch, next_ob_batch, terminal_batch = self.controller.sample_data(self.params['train_batch_size'])
            self.controller.train(ob_batch, expert_ac_batch)


    # def do_relabel_with_expert(self, paths):
    #     print("Relabelling collected observations with labels from an expert policy...")
    #
    #     for i in range(len(paths)):
    #         acs = self.expert_policy.get_action(paths[i]["observation"])
    #         paths[i]["action"] = acs
    #
    #     return paths
