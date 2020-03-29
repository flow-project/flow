import numpy as np
import tensorflow
from tensorflow import keras
import tensorflow as tf
from utils import *
from flow.controllers.base_controller import BaseController
from replay_buffer import ReplayBuffer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import *



class ImitatingController(BaseController):
    # Implementation in Keras just for testing

    def __init__(self, veh_id, sess, action_dim, obs_dim, num_layers, size, learning_rate, replay_buffer_size, training = True, policy_scope='policy_vars', car_following_params=None, time_delay=0.0, noise=0, fail_safe=None):

        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise)
        self.sess = sess
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_layers = num_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.model = Sequential()
        self.build_network()



        if self.training:
            self.replay_buffer = ReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = None

    def build_network(self):
        self.model.add(Dense(self.size, input_dim=self.obs_dim, activation='tanh'))
        for _ in range(self.num_layers):
            self.model.add(Dense(self.size, activation='relu'))
        # No activation
        self.model.add(Dense(self.action_dim))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    def train(self, observation_batch, action_batch):
        assert(self.training, "Policy must be trainable")

        print("OBS NAN CHECK: ", np.any(np.isnan(observation_batch)))
        assert (not np.any(np.isnan(action_batch))), "TRAIN ERROR ACTION NAN"

        action_batch = action_batch.reshape(action_batch.shape[0], self.action_dim)
        history = self.model.fit(observation_batch, action_batch)

    def get_accel_from_observation(self, observation):
        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        if len(observation.shape)<=1:
            observation = observation[None]
        ret_val = self.model.predict(observation)

        return ret_val

    def get_accel(self, env):
        # network expects an array of arrays (matrix); if single observation (no batch), convert to array of arrays
        observation = env.get_state()
        return self.get_accel_from_observation(observation)

    def add_to_replay_buffer(self, rollout_list):
        self.replay_buffer.add_rollouts(rollout_list)

    def sample_data(self, batch_size):
        return self.replay_buffer.sample_batch(batch_size)
