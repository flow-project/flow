import tensorflow as tf
import os
import numpy as np
import math

""" Class agnostic helper functions """

def sample_trajectory_singleagent(env, vehicle_id, controller, expert_controller, max_trajectory_length):
    """
    Samples a trajectory for a given vehicle using the actions prescribed by specified controller.
    Args:
        env: environment
        vehicle_id: id of the vehicle that is being controlled/tracked during trajectory
        controller: subclass of BaseController, decides actions taken by vehicle
        expert_controller: subclass of BaseController, "expert" for imitation learning
        max_trajectory_length: maximum steps in a trajectory
    Returns:
        Dictionary of numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

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
        terminate_rollout = (traj_length == max_trajectory_length) or done
        terminals.append(terminate_rollout)

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals)


def sample_trajectory_multiagent(env, vehicle_ids, controllers, expert_controllers, max_trajectory_length):
    """
    Samples a trajectory for a given set of vehicles using the actions prescribed by specified controller.

    Args:
        env: environment
        vehicle_ids: id of the vehicle that is being controlled/tracked during trajectory
        controllers: subclass of BaseController, decides actions taken by vehicle
        expert_controllers: subclass of BaseController, "expert" for imitation learning
        max_trajectory_length: maximum steps in a trajectory

    Returns:
        Dictionary of numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

    print("COLLECTING CONTROLLER: ", controllers[0])
    print("EXPERT CONTROLLER: ", expert_controllers[0])
    observation_dict = env.reset()

    for vehicle_id in vehicle_ids:
        assert vehicle_id in env.k.vehicle.get_ids(), "Vehicle ID not in env!"

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:
        rl_actions = dict()
        invalid_expert_action = False
        expert_action_dict = dict()

        for i in range(len(vehicle_ids)):
            vehicle_id = vehicle_ids[i]
            controller = controllers[i]
            expert_controller = expert_controllers[i]

            action = controller.get_action(env)

            if type(action) == np.ndarray:
                action = action.flatten()[0]

            expert_action = expert_controller.get_action(env)
            expert_action_dict[vehicle_id] = expert_action

            if (expert_action is None or math.isnan(expert_action)):
                invalid_expert_action = True

            rl_actions[vehicle_id] = action

        if invalid_expert_action:
            # invalid action in rl_actions, so default control to SUMO
            observations_dict, reward_dict, done_dict, _ = env.step(None)
            traj_length += 1
            terminate_rollout = traj_length == max_trajectory_length or done_dict['__all__']
            if terminate_rollout:
                break
            continue

        for vehicle_id in vehicle_ids:
            observations.append(observation_dict[vehicle_id])
            actions.append(rl_actions[vehicle_id])
            expert_actions.append(expert_action_dict[vehicle_id])

        observation_dict, reward_dict, done_dict, _ = env.step(rl_actions)
        terminate_rollout = done_dict['__all__'] or (traj_length == max_trajectory_length)

        for vehicle_id in vehicle_ids:
            next_observations.append(observation_dict[vehicle_id])
            rewards.append(reward_dict[vehicle_id])
            terminals.append(terminate_rollout)

        traj_length += 1

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals)


def sample_trajectories(env, vehicle_ids, controllers, expert_controllers, min_batch_timesteps, max_trajectory_length, multiagent):
    """
    Samples trajectories to collect at least min_batch_timesteps steps in the environment

    Args:
        env: environment
        vehicle_id: id of vehicle being tracked/controlled
        controller: subclass of BaseController, decides actions taken by vehicle
        expert_controller: subclass of BaseController, "expert" for imitation learning
        min_batch_timesteps: minimum number of environment steps to collect
        max_trajectory_length: maximum steps in a trajectory

    Returns:
        List of rollout dictionaries, total steps taken by environment
    """
    total_envsteps = 0
    trajectories = []

    while total_envsteps < min_batch_timesteps:

        if multiagent:
            trajectory = sample_trajectory_multiagent(env, vehicle_ids, controllers, expert_controllers, max_trajectory_length)
        else:
            trajectory = sample_trajectory_singleagent(env, vehicle_ids[0], controllers[0], expert_controllers[0], max_trajectory_length)

        trajectories.append(trajectory)

        traj_env_steps = len(trajectory["rewards"]) / len(vehicle_ids)
        total_envsteps += traj_env_steps

    return trajectories, total_envsteps

def sample_n_trajectories(env, vehicle_ids, controllers, expert_controllers, n, max_trajectory_length, multiagent):
    """
    Collects a fixed number of trajectories.

    Args:
        env: environment
        vehicle_id: id of vehicle being tracked/controlled
        controller: subclass of BaseController, decides actions taken by vehicle
        expert_controller: subclass of BaseController, "expert" for imitation learning
        n: number of trajectories to collect
        max_trajectory_length: maximum steps in a trajectory

    Returns:
        List of rollout dictionaries

    """
    trajectories = []
    for _ in range(n):

        if multiagent:
            trajectory = sample_trajectory_multiagent(env, vehicle_ids, controllers, expert_controllers, max_trajectory_length)
        else:
            trajectory = sample_trajectory_singleagent(env, vehicle_ids[0], controllers[0], expert_controllers[0], max_trajectory_length)

        trajectories.append(trajectory)

    return trajectories


def traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals):
    """
    Collects individual observation, action, expert_action, rewards, next observation, terminal arrays into a single rollout dictionary
    """
    return {"observations" : np.array(observations, dtype=np.float32),
            "actions" : np.array(actions, dtype=np.float32),
            "expert_actions": np.array(expert_actions, dtype=np.float32),
            "rewards" : np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "terminals": np.array(terminals, dtype=np.float32)}


def unpack_rollouts(rollouts_list):
    """
        Convert list of rollout dictionaries to individual observation, action, rewards, next observation, terminal arrays
        rollouts: list of rollout dictionaries, rollout dictionary: dictionary with keys "observations", "actions", "rewards", "next_observations", "is_terminals"
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

def build_neural_net(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedfoward neural network for action prediction

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of pass through Neural Network
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
