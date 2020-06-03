import time
from collections import OrderedDict
import pickle
import numpy as np
import gym
import os
import tensorflow as tf
from utils import *
from flow.utils.registry import make_create_env
from flow.controllers.imitation_learning.imitating_controller import ImitatingController
from flow.controllers.imitation_learning.imitating_network import ImitatingNetwork
from flow.controllers.imitation_learning.utils_tensorflow import *
from flow.controllers.imitation_learning.keras_utils import *
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.params import SumoCarFollowingParams

class Trainer(object):
    """
    Class to initialize and run training for imitation learning (with DAgger)
    """

    def __init__(self, params, submodule):
        """
        Parameters
        __________
        params: dict
            Dictionary of parameters used to run imitation learning
        submodule: Module
            Python module for file containing flow_params
        """

        # get flow params
        self.flow_params = submodule.flow_params

        # setup parameters for training
        self.params = params
        self.sess = create_tf_session()

        # environment setup
        create_env, _ = make_create_env(self.flow_params)
        self.env = create_env()

        # vehicle setup
        self.multiagent = self.params['multiagent'] # multiagent or singleagent env

        if not self.multiagent and self.env.action_space.shape[0] > 1:
            # use sorted rl ids if the method exists (e.g.. singlagent straightroad)
            try:
                self.vehicle_ids = self.env.get_sorted_rl_ids()
            except:
                self.vehicle_ids = self.k.vehicle.get_rl_ids()
        else:
            # use get_rl_ids if sorted_rl_ids doesn't exist
            self.vehicle_ids = self.env.k.vehicle.get_rl_ids()

        # neural net setup
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.params['action_dim'] = action_dim
        self.params['obs_dim'] = obs_dim

        # initialize neural network class and tf variables
        self.action_network = ImitatingNetwork(self.sess, self.params['action_dim'], self.params['obs_dim'], self.params['fcnet_hiddens'], self.params['replay_buffer_size'], stochastic=self.params['stochastic'], variance_regularizer=self.params['variance_regularizer'], load_model=self.params['load_imitation_model'], load_path=self.params['load_imitation_path'], tensorboard_path=self.params['tensorboard_path'])


        # controllers setup
        v_des = self.params['v_des'] # for FollowerStopper
        car_following_params = SumoCarFollowingParams()
        self.controllers = dict()
        # initialize controllers: save in a dictionary to avoid re-initializing a controller for a vehicle
        for vehicle_id in self.vehicle_ids:
            expert = FollowerStopper(vehicle_id, car_following_params=car_following_params, v_des=v_des)
            imitator = ImitatingController(vehicle_id, self.action_network, self.multiagent, car_following_params=car_following_params)
            self.controllers[vehicle_id] = (imitator, expert)


    def run_training_loop(self, n_iter):
        """
        Trains imitator for n_iter iterations (each iteration collects new trajectories to put in replay buffer)

        Parameters
        __________
        n_iter :
            intnumber of iterations to execute training
        """

        # init vars at beginning of training
        # number of environment steps taken throughout training
        self.total_envsteps = 0

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            if itr == 0:
                # first iteration is behavioral cloning
                training_returns = self.collect_training_trajectories(itr, self.params['init_batch_size'])
            else:
                # other iterations use DAgger (trajectories collected by running imitator policy)
                training_returns = self.collect_training_trajectories(itr, self.params['batch_size'])

            paths, envsteps_this_batch = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer in neural network class
            self.action_network.add_to_replay_buffer(paths)

            # train controller
            self.train_controller()

    def collect_training_trajectories(self, itr, batch_size):
        """
        Collect (state, action, reward, next_state, terminal) tuples for training

        Parameters
        __________
        itr: int
            iteration of training during which function is called. Used to determine whether to run behavioral cloning or DAgger
        batch_size: int
            number of tuples to collect
        Returns
        _______
        paths: list
            list of trajectories
        envsteps_this_batch: int
            the sum over the numbers of environment steps in paths (total number of env transitions in trajectories collected)
        """

        print("\nCollecting data to be used for training...")
        max_decel = self.flow_params['env'].additional_params['max_decel']
        trajectories, envsteps_this_batch = sample_trajectories(self.env, self.controllers, self.action_network, batch_size, self.params['ep_len'], self.multiagent, use_expert=itr==0, v_des=self.params['v_des'], max_decel=max_decel)

        return trajectories, envsteps_this_batch

    def train_controller(self):
        """
        Trains controller for specified number of steps, using data sampled from replay buffer; each step involves running optimizer (i.e. Adam) once
        """

        print("Training controller using sampled data from replay buffer...")
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # sample data from replay buffer
            ob_batch, ac_batch, expert_ac_batch = self.action_network.sample_data(self.params['train_batch_size'])
            # train network on sampled data
            self.action_network.train(ob_batch, expert_ac_batch)

    def evaluate_controller(self, num_trajs = 10):
        """
        Evaluates a trained imitation controller on similarity with expert with respect to action taken and total reward per rollout.

        Parameters
        __________
        num_trajs: int
            number of trajectories to evaluate performance on
        """

        print("\n\n********** Evaluation ************ \n")


        # collect imitator driven trajectories (along with corresponding expert actions)
        trajectories = sample_n_trajectories(self.env, self.controllers, self.action_network, num_trajs, self.params['ep_len'], self.multiagent, False, v_des=self.params['v_des'])

        # initialize metrics
        total_imitator_steps = 0  # total number of environment steps taken across the n trajectories
        average_imitator_reward_per_rollout = 0 # average reward per rollout achieved by imitator

        action_errors = np.array([]) # difference in action (acceleration) taken between expert and imitator
        average_action_expert = 0 # average action taken, across all timesteps, by expert (used to compute % average)
        average_action_imitator = 0 # average action taken, across all timesteps, by imitator (used to compute % average)

        # compare actions taken in each step of trajectories (trajectories are controlled by imitator)
        for traj_tuple in trajectories:
            traj = traj_tuple[0]
            traj_len = traj_tuple[1]

            imitator_actions = traj['actions']
            expert_actions = traj['expert_actions']

            average_action_expert += np.sum(expert_actions)
            average_action_imitator += np.sum(imitator_actions)

            # use RMSE as action error metric
            action_error = (np.linalg.norm(imitator_actions - expert_actions)) / len(imitator_actions)
            action_errors = np.append(action_errors, action_error)

            total_imitator_steps += traj_len
            average_imitator_reward_per_rollout += np.sum(traj['rewards'])

        # compute averages for metrics
        average_imitator_reward_per_rollout = average_imitator_reward_per_rollout / len(trajectories)

        average_action_expert = average_action_expert / total_imitator_steps

        # collect expert driven trajectories (these trajectories are only used to compare average reward per rollout)
        expert_trajectories = sample_n_trajectories(self.env, self.controllers, self.action_network, num_trajs, self.params['ep_len'], self.multiagent, True, v_des=self.params['v_des'])

        # initialize metrics
        total_expert_steps = 0
        average_expert_reward_per_rollout = 0

        # compare reward accumulated in trajectories collected via expert vs. via imitator
        for traj_tuple in expert_trajectories:
            traj = traj_tuple[0]
            traj_len = traj_tuple[1]
            total_expert_steps += traj_len
            average_expert_reward_per_rollout += np.sum(traj['rewards'])

        average_expert_reward_per_rollout = average_expert_reward_per_rollout / len(expert_trajectories)

        # compute percent errors (using expert values as 'ground truth')
        percent_error_average_reward = (np.abs(average_expert_reward_per_rollout - average_imitator_reward_per_rollout) / average_expert_reward_per_rollout) * 100

        percent_error_average_action = (np.abs(np.mean(action_errors)) / np.abs(average_action_expert)) * 100

        # Print results
        print("\nAverage reward per rollout, expert: ", average_expert_reward_per_rollout)
        print("Average reward per rollout, imitator: ", average_imitator_reward_per_rollout)
        print("% Difference, average reward per rollout: ", percent_error_average_reward, "\n")


        print(" Average RMSE action error per rollout: ", np.mean(action_errors))
        print("Average Action Taken by Expert: ", average_action_expert)
        print("% Action Error: ", percent_error_average_action, "\n")
        print("Total imitator steps: ", total_imitator_steps)
        print("Total expert steps: ", total_expert_steps)

    def learn_value_function(self, num_samples, num_iterations, num_grad_steps):
        """
        Learn the value function under imitation policy.
        Parameters
        __________
        num_samples: number of environment transition samples to collect to learn from
        num_iterations: number of iterations to relabel data, and train
        num_grad_steps: number of gradient steps per training iteration

        Returns
        _______
        Value function neural net
        """
        # init value function neural net
        vf_net = build_neural_net_deterministic(self.params['obs_dim'], 1, self.params['fcnet_hiddens'])
        vf_net.compile(loss='mean_squared_error', optimizer = 'adam')

        max_decel = self.flow_params['env'].additional_params['max_decel']
        # collect trajectory samples to train on
        trajectories, envsteps_this_batch = sample_trajectories(self.env, self.controllers, self.action_network,
                                                                num_samples, self.params['ep_len'], self.multiagent,
                                                                use_expert=False, v_des=self.params['v_des'],
                                                                max_decel=max_decel)

        # combine trajectories into one
        observations = np.concatenate([traj['observations'] for traj in trajectories])
        rewards = np.concatenate([traj['rewards'] for traj in trajectories])
        next_observations = np.concatenate([traj['next_observations'] for traj in trajectories])

        # iterate over data multiple times (labels change every iteration)
        for _ in range(num_iterations):
            # form labels
            next_state_value_preds = vf_net.predict(next_observations).flatten()
            next_state_value_preds[np.isnan(next_state_value_preds)] = 0
            labels = rewards + next_state_value_preds
            vf_net.fit(observations, labels, verbose=0)

        return vf_net



    def save_controller_for_PPO(self):
        """
        Build a model, with same policy architecture as imitation network, to run PPO, copy weights from imitation, and save this model.

        """

        vf_net = self.learn_value_function(self.params['vf_batch_size'], self.params['num_vf_iters'], self.params['num_agent_train_steps_per_iter'])

        input = tf.keras.layers.Input(self.action_network.model.input.shape[1].value)
        curr_layer = input

        # number of hidden layers
        num_layers = len(self.action_network.model.layers) - 2

        # build layers for policy
        for i in range(num_layers):
            size = self.action_network.model.layers[i + 1].output.shape[1].value
            activation = tf.keras.activations.serialize(self.action_network.model.layers[i + 1].activation)
            curr_layer = tf.keras.layers.Dense(size, activation=activation, name="policy_hidden_layer_{}".format(i + 1))(curr_layer)
        output_layer_policy = tf.keras.layers.Dense(self.action_network.model.output.shape[1].value, activation=None, name="policy_output_layer")(curr_layer)

        # build layers for value function
        curr_layer = input
        for i in range(num_layers):
            size = self.params['fcnet_hiddens'][i]
            curr_layer = tf.keras.layers.Dense(size, activation="tanh", name="vf_hidden_layer_{}".format(i+1))(curr_layer)
        output_layer_vf = tf.keras.layers.Dense(1, activation=None, name="vf_output_layer")(curr_layer)

        ppo_model = tf.keras.Model(inputs=input, outputs=[output_layer_policy, output_layer_vf], name="ppo_model")

        # set the policy weights to those learned from imitation
        for i in range(num_layers):
            policy_layer = ppo_model.get_layer(name="policy_hidden_layer_{}".format(i + 1))
            policy_layer.set_weights(self.action_network.model.layers[i + 1].get_weights())
        policy_output = ppo_model.get_layer("policy_output_layer")
        policy_output.set_weights(self.action_network.model.layers[-1].get_weights())

        # set value function weights to those learned
        num_vf_layers = len(vf_net.layers) - 2
        for i in range(num_vf_layers):
            vf_layer = ppo_model.get_layer('vf_hidden_layer_{}'.format(i + 1))
            vf_layer.set_weights(vf_net.layers[i + 1].get_weights())
        vf_output = ppo_model.get_layer("vf_output_layer")
        vf_output.set_weights(vf_net.layers[-1].get_weights())


        # save the model (as a h5 file)
        ppo_model.save(self.params['PPO_save_path'])


    def save_controller_network(self):
        """
        Saves a keras tensorflow model to the specified path given in the command line params. Path must end with .h5.
        """
        print("Saving tensorflow model to: ", self.params['save_path'])
        self.action_network.save_network(self.params['save_path'])
