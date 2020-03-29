import tensorflow as tf
import os
import numpy as np
import math

# class agnostic helper functions

def sample_trajectory(env, vehicle_id, controller, expert_controller, max_trajectory_length):

    print("COLLECTING CONTROLLER: ", controller)
    print("EXPERT CONTROLLER: ", expert_controller)
    observation = env.reset()

    assert vehicle_id in env.k.vehicle.get_ids(), "Vehicle ID not in env!"

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:
        action = controller.get_action(env)

        if type(action) == np.ndarray:
            action = action.flatten()[0]

        expert_action = expert_controller.get_action(env)
        if (expert_action is None or math.isnan(expert_action)):
            observation, reward, done, _ = env.step(action)
            traj_length += 1
            terminate_rollout = traj_length == max_trajectory_length or done
            if terminate_rollout:
                break
            continue

        observations.append(observation)
        actions.append(action)
        expert_actions.append(expert_action)
        observation, reward, done, _ = env.step(action)

        traj_length += 1
        next_observations.append(observation)
        rewards.append(reward)
        terminate_rollout = traj_length == max_trajectory_length or done
        terminals.append(terminate_rollout)

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals)


def sample_trajectories(env, vehicle_id, controller, expert_controller, min_batch_timesteps, max_trajectory_length):
    total_envsteps = 0
    trajectories = []

    while total_envsteps < min_batch_timesteps:
        trajectory = sample_trajectory(env, vehicle_id, controller, expert_controller, max_trajectory_length)
        trajectories.append(trajectory)

        traj_env_steps = len(trajectory["rewards"])
        total_envsteps += traj_env_steps

    return trajectories, total_envsteps

def traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals):
    return {"observations" : np.array(observations, dtype=np.float32),
            "actions" : np.array(actions, dtype=np.float32),
            "expert_actions": np.array(expert_actions, dtype=np.float32),
            "rewards" : np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "terminals": np.array(terminals, dtype=np.float32)}

def unpack_rollouts(rollouts_list):
    """
        Convert list of rollout dictionaries to individual observation, action, rewards, next observation, terminal arrays
        rollouts: list of rollout dictionaries
        rollout dictionary: dictionary with keys "observations", "actions", "rewards", "next_observations", "is_terminals"
        return separate np arrays of observations, actions, rewards, next_observations, and is_terminals
    """
    observations = np.concatenate([rollout["observations"] for rollout in rollouts_list])
    actions = np.concatenate([rollout["actions"] for rollout in rollouts_list])
    expert_actions = np.concatenate([rollout["expert_actions"] for rollout in rollouts_list])
    rewards = np.concatenate([rollout["rewards"] for rollout in rollouts_list])
    next_observations = np.concatenate([rollout["next_observations"] for rollout in rollouts_list])
    terminals = np.concatenate([rollout["terminals"] for rollout in rollouts_list])

    return observations, actions, expert_actions, rewards, next_observations, terminals


# Below are tensorflow related functions

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedfoward neural net

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(output_placeholder, size, activation=activation)
        output_placeholder = tf.layers.dense(output_placeholder, output_size, activation=output_activation)
    return output_placeholder

def create_tf_session():
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    return sess
