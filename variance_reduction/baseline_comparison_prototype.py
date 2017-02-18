import numpy as np
import tensorflow as tf

import gym

"""
Adapted from:
https://gist.github.com/awjuliani/86ae316a231bceb96a3e2ab3ac8e646a#file-rl-tutorial-2-ipynb
"""

env = gym.make('Walker2d-v1')  # Requires Mujoco

SHOW_THRESH = 0
SHOW_EVERY = 300

# hyperparameters for main function approximator
H = 32  # number of hidden layer neurons
batch_size = 300  # every how many episodes to do a param update?
learning_rate = 1e-2  # feel free to play with this to train faster or more
# stably.
gamma = 0.99  # discount factor for reward

# hyperparameters for baselines
H_BASELINE = 32  # number of hidden layer neurons

D = env.observation_space.shape[0]
Out = env.action_space.shape[0]
print("Size of obs and action spaces: ", D, Out)

# goal_reward = 200
goal_reward = 1000

# ------------------- Program start -------------------------------------------

tf.reset_default_graph()

# This defines the network as it goes from taking an observation of the
# environment to
# giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
# layer1 = tf.matmul(observations, W1)
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, Out],
                     initializer=tf.truncated_normal_initializer(
                         stddev=0.01))
W3 = tf.get_variable("W3", shape=[H, Out],
                     initializer=tf.truncated_normal_initializer(
                         mean=0, stddev=0.01))
output = tf.concat(1, [tf.matmul(layer1, W2), tf.matmul(layer1, W3)])

# From here we define the parts of the network needed for learning a good
# policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, Out], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

score_old = tf.placeholder(tf.float32, [None, Out * 2], name="score_old")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that
# didn't less likely.
# Log likelihood: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
# #Likelihood_function
means = output[0][:Out]
log_stds = output[0][Out:]
zs = (input_y - means) / tf.exp(log_stds)
loglik_new = - tf.reduce_sum(log_stds,
                             reduction_indices=-1) - 0.5 * tf.reduce_sum(
    tf.square(zs), reduction_indices=-1) - 0.5 * Out * np.log(2 * np.pi)

means_old = score_old[0][:Out]
log_stds_old = score_old[0][Out:]
zs_old = (input_y - means_old) / tf.exp(log_stds_old)
loglik_old = - tf.reduce_sum(log_stds_old,
                             reduction_indices=-1) - 0.5 * tf.reduce_sum(
    tf.square(zs_old), reduction_indices=-1) - 0.5 * Out * np.log(2 * np.pi)

likratio = tf.exp(loglik_new - loglik_old)
loss = -tf.reduce_mean(likratio * advantages)
newGrads = tf.gradients(loss, tvars)

# TODO(cathywu) check that the gradient update code is alright for continuous
# problems.
# Once we have collected a series of gradients from multiple episodes,
# we apply them. We don't just apply gradients after every episode in order
# to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
# Placeholders to send the final gradients through when we update.
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
W3Grad = tf.placeholder(tf.float32, name="batch_grad3")
batchGrad = [W1Grad, W2Grad, W3Grad]
# TODO(cathywu) How does tf figure out which variables to update in tvars?
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

# TODO(cathywu) function approximator for V(s) baseline
Q_hats = tf.placeholder(tf.float32, [None, 1], name="Q_hats")
baseline_W1 = tf.get_variable("baseline_W1", shape=[D, H_BASELINE],
                              initializer=tf.contrib.layers.xavier_initializer())
baseline_layer1 = tf.matmul(observations, baseline_W1)
baseline_W2 = tf.get_variable("baseline_W2", shape=[H_BASELINE, 1],
                              initializer=tf.truncated_normal_initializer(
                                  stddev=0.01))
vBaseline = tf.matmul(baseline_layer1, baseline_W2)
baseline_loss = -tf.reduce_sum(tf.square(vBaseline - Q_hats))
# FIXME(cathywu) use of tvars here? Trainable variables?
baseline_newGrads = tf.gradients(baseline_loss, tvars)

# TODO(cathywu) function approximator for \sum_i Q(s,-i) baseline
# TODO(cathywu) How to programmatically generate tensors and refer to them?
# aBaseline =

# Normalize the observations, important because the distribution of the
# observations changes over time.
# TODO(cathywu) a better way to do this is to normalize wrt rolling statistics.
obs = []
done = True
for _ in range(1000):
    if done:
        observation = env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())
    obs.append(observation)
obs_mean = np.mean(np.vstack(obs), axis=0)
obs_std = np.std(np.vstack(obs), axis=0)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
epxs, epys, discounted_eprs = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
total_episodes = 30000
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    # Obtain an initial observation of the environment
    observation = env.reset()

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    x = (np.reshape(observation, [1, D]) - obs_mean) / obs_std
    old_score = output.eval(feed_dict={observations: x})

    while episode_number < total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if (
                            reward_sum / batch_size > SHOW_THRESH or rendering == True) and episode_number % SHOW_EVERY == 0:
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.
        x = (np.reshape(observation, [1, D]) - obs_mean) / obs_std

        # Run the policy network and get an action to take.
        tfscore = sess.run(output, feed_dict={observations: x})
        action = np.exp(tfscore[0][Out:]) * np.random.normal(size=Out) + \
                 tfscore[0][:Out]
        # print("Action:", tfscore[0][:Out], np.exp(tfscore[0][Out:]), action)

        xs.append(x)  # observation
        ys.append(action)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # record reward (has to be done after we call step() to get reward
        # for previous action)
        drs.append(reward)

        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients,
            # and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            # reset array memory
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)

            # Variance
            variance = np.mean(np.sum(np.square(discounted_epr))) - np.square(
                np.mean(discounted_epr))
            # size the rewards to be unit normal (helps control the gradient
            #  estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            variance_avgbaseline = np.mean(
                np.sum(np.square(discounted_epr))) - np.square(
                np.mean(discounted_epr))

            # print("Variance: ", variance, variance_avgbaseline)

            epxs.append(epx)
            epys.append(epy)
            discounted_eprs.append(discounted_epr)

            # If we have completed enough episodes, then update the policy
            # network with our gradients.
            if episode_number % batch_size == 0:
                batch_Qhats = np.vstack(discounted_eprs)
                batch_obs = np.vstack(epxs)
                batch_actions = np.vstack(epys)
                # TODO(cathywu) compute V(s) here using Q_hat, s



                # TODO(cathywu) compute \sum_i Q(s,-i) here using Q_hat, s, -i

                # Get the gradient for each episode, and save it in the
                # gradBuffer
                # TODO(cathywu) compute gradients here
                for epx, epy, discounted_epr in zip(epxs, epys,
                                                    discounted_eprs):
                    tGrad = sess.run(newGrads,
                                     feed_dict={observations: epx, input_y: epy,
                                                advantages: discounted_epr,
                                                score_old: old_score})
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] += grad

                # TODO(cathywu) compute gradients with V baseline

                # TODO(cathywu) compute gradients with action baseline

                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
                                                 W2Grad: gradBuffer[1],
                                                 W3Grad: gradBuffer[2]})
                old_score = output.eval(feed_dict={observations: x})

                # Clear buffers and temporary batch storage lists
                epxs, epys, discounted_eprs = [], [], []
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each
                # batch of episodes.
                running_reward = reward_sum if running_reward is None else \
                    running_reward * 0.99 + reward_sum * 0.01
                print("[{0:d}] Average reward for episode: {1:6.2f}. Total "
                      "average reward {2:6.2f}.".format(episode_number,
                                                        reward_sum / batch_size,
                                                        running_reward / batch_size))

                if reward_sum / batch_size > goal_reward:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')
