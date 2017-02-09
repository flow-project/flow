import numpy as np
import tensorflow as tf

import gym

"""
Adapted from:
https://gist.github.com/awjuliani/86ae316a231bceb96a3e2ab3ac8e646a#file-rl-tutorial-2-ipynb
"""

# env = gym.make('CartPole-v0')
env = gym.make('Walker2d-v1')  # Requires Mujoco

env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 5:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()

# hyperparameters
H = 10  # number of hidden layer neurons
batch_size = 5  # every how many episodes to do a param update?
learning_rate = 1e-2  # feel free to play with this to train faster or more
# stably.
gamma = 0.99  # discount factor for reward

# D = 4  # input dimensionality
D = env.observation_space.shape[0]
Out = env.action_space.shape[0]
print("Size of obs and action spaces: ", D, Out)

# goal_reward = 200
goal_reward = 1000

tf.reset_default_graph()

# This defines the network as it goes from taking an observation of the
# environment to
# giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, Out],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# From here we define the parts of the network needed for learning a good
# policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, Out], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that
# didn't less likely.
# TODO what's a reasonable reward function?
loglik = tf.log(
    input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes,
# we apply them. We don't just apply gradients after every episode in order
# to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
# Placeholders to send the final gradients through when we update.
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
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

    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x})
        # action = 1 if np.random.uniform() < tfprob else 0
        # action = tfprob
        action = 5 * (tfprob - 0.5)
        print(action)

        xs.append(x)  # observation
        # y = 1 if action == 0 else 0  # a "fake label"
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
            # and  rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            # reset array memory
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient
            #  estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,
                             feed_dict={observations: epx, input_y: epy,
                                        advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy
            # network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0],
                                                 W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # TODO compute Q_hat here (leaving off here)

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
