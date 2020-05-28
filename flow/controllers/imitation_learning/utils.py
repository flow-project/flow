import tensorflow as tf
import os
import numpy as np
import math
from flow.core.params import SumoCarFollowingParams
from imitating_controller import ImitatingController
from imitating_network import ImitatingNetwork
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper
from flow.core.rewards import *

""" Class agnostic helper functions """

def sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des, max_decel):
    """
    Samples a single trajectory from a singleagent environment.
    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    max_trajectory_length: int
        maximum steps in a trajectory
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    v_des: float
        v_des parameter for follower-stopper
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.
    Returns
    _______
    dict
        Dictionary of numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

    # reset and initialize arrays to store trajectory
    observation = env.reset()

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:

        # update vehicle ids: if multidimensional action space, check if env has a sorted_rl_ids method
        if env.action_space.shape[0] > 1:
            try:
                vehicle_ids = env.get_sorted_rl_ids()
            except:
                vehicle_ids = env.k.vehicle.get_rl_ids()
        else:
            vehicle_ids = env.k.vehicle.get_rl_ids()

        # no RL actions if no RL vehicles
        if len(vehicle_ids) == 0:
            observation, reward, done, _ = env.step(None)
            if done:
                break
            continue

        # init controllers if any of vehicle ids are new
        # there could be multiple vehicle ids if they all share one state but have different actions
        car_following_params = SumoCarFollowingParams()

        for vehicle_id in vehicle_ids:
            if vehicle_id not in set(controllers.keys()):
                expert = FollowerStopper(vehicle_id, car_following_params=car_following_params, v_des=v_des)
                imitator = ImitatingController(vehicle_id, action_network, False, car_following_params=car_following_params)
                controllers[vehicle_id] = (imitator, expert)


        # get the actions given by controllers
        action_dim = env.action_space.shape[0]
        rl_actions = []
        actions_expert = []

        invalid_expert_action = False
        for i in range(action_dim):
            # if max number of RL vehicles is not reached, insert dummy values
            if i >= len(vehicle_ids):
                # dummy value is -2 * max_decel
                ignore_accel = -2 * max_decel
                rl_actions.append(ignore_accel)
                actions_expert.append(ignore_accel)
            else:
                imitator = controllers[vehicle_ids[i]][0]
                expert = controllers[vehicle_ids[i]][1]

                expert_action = expert.get_action(env)
                # catch invalid expert actions
                if (expert_action is None or math.isnan(expert_action)):
                    invalid_expert_action = True

                actions_expert.append(expert_action)

                if use_expert:
                    if traj_length == 0 and i == 0:
                        print("Controller collecting trajectory: ", type(expert))
                    rl_actions.append(expert_action)
                else:
                    if traj_length == 0 and i == 0:
                        print("Controller collecting trajectory: ", type(imitator))
                    imitator_action = imitator.get_action(env)
                    rl_actions.append(imitator_action)


        # invalid action in rl_actions; default to Sumo, ignore sample
        if None in rl_actions or np.nan in rl_actions:
            observation, reward, done, _ = env.step(None)
            terminate_rollout = traj_length == max_trajectory_length or done
            if terminate_rollout:
                break
            continue
        # invalid expert action (if rl_actions is expert actions then this would have been caught above))
        if not use_expert and invalid_expert_action:
            # throw away sample, but step according to rl_actions
            observation, reward, done, _ = env.step(rl_actions)
            terminate_rollout = traj_length == max_trajectory_length or done
            if terminate_rollout:
                break
            continue

        # update collected data
        observations.append(observation)
        actions.append(rl_actions)
        expert_actions.append(actions_expert)
        observation, reward, done, _ = env.step(rl_actions)

        traj_length += 1
        next_observations.append(observation)
        rewards.append(reward)
        terminate_rollout = (traj_length == max_trajectory_length) or done
        terminals.append(terminate_rollout)

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals), traj_length


def sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des):
    """
    Samples a single trajectory from a multiagent environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    max_trajectory_length: int
        maximum steps in a trajectory
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    v_des: float
        v_des parameter for follower-stopper
    Returns
    _______
    dict
        Dictionary of numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

    observation_dict = env.reset()

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:

        vehicle_ids = list(observation_dict.keys())
        # add nothing to replay buffer if no vehicles
        if len(vehicle_ids) == 0:
            observation_dict, reward, done, _ = env.step(None)
            if done['__all__']:
                break
            continue

        # actions taken by collecting controller
        rl_actions = dict()
        invalid_expert_action = False
        # actions taken by expert
        expert_action_dict= dict()

        for i in range(len(vehicle_ids)):
            vehicle_id = vehicle_ids[i]

            if vehicle_id not in set(controllers.keys()):
                car_following_params = SumoCarFollowingParams()

                expert = FollowerStopper(vehicle_id, car_following_params=car_following_params, v_des=v_des)
                imitator = ImitatingController(vehicle_id, action_network, True, car_following_params=car_following_params)
                controllers[vehicle_id] = (imitator, expert)

            expert_controller = controllers[vehicle_id][1]
            if use_expert:
                controller = expert_controller
            else:
                controller = controllers[vehicle_id][0]

            if traj_length == 0 and i == 0:
                print("Controller collecting trajectory: ", controller)

            action = controller.get_action(env)

            # action should be a scalar acceleration
            if type(action) == np.ndarray:
                action = action.flatten()[0]

            expert_action = expert_controller.get_action(env)
            expert_action_dict[vehicle_id] = expert_action

            if (expert_action is None or math.isnan(expert_action)):
                invalid_expert_action = True

            rl_actions[vehicle_id] = action

        if invalid_expert_action:
            # invalid action in rl_actions, so default control to SUMO
            observation_dict, reward_dict, done_dict, _ = env.step(None)
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
            next_observations.append(observation_dict.get(vehicle_id, None))
            rewards.append(reward_dict.get(vehicle_id, 0))
            terminals.append(terminate_rollout)

        traj_length += 1

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals), traj_length


def sample_trajectories(env, controllers, action_network, min_batch_timesteps, max_trajectory_length, multiagent, use_expert, v_des=15, max_decel=4.5):
    """
    Samples trajectories from environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    min_batch_timesteps: int
        minimum number of env transitions to collect
    max_trajectory_length: int
        maximum steps in a trajectory
    multiagent: bool
        if True, env is a multiagent env
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    v_des: float
        v_des parameter for follower-stopper
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.

    Returns
    _______
    dict, int
        Dictionary of trajectory numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
        Total number of env transitions seen over trajectories
    """
    total_envsteps = 0
    trajectories = []

    while total_envsteps < min_batch_timesteps:

        if multiagent:
            trajectory, traj_length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des)
        else:
            trajectory, traj_length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des, max_decel)

        trajectories.append(trajectory)

        total_envsteps += traj_length

    return trajectories, total_envsteps

def sample_n_trajectories(env, controllers, action_network, n, max_trajectory_length, multiagent, use_expert, v_des=15, max_decel=4.5):
    """
    Samples n trajectories from environment.

    Parameters
    __________
    env: gym.Env
        environment
    controllers: dict
        Dictionary of 2-tuples (Imitating_Controller, Expert_Controller), with keys of vehicle_ids
    action_network: ImitatingNetwork
        ImitatingNetwork class containing neural net for action prediction
    n: int
        number of trajectories to collect
    max_trajectory_length: int
        maximum steps in a trajectory
    multiagent: bool
        if True, env is a multiagent env
    use_expert: bool
        if True, trajectory is collected using expert policy (for behavioral cloning)
    v_des: float
        v_des parameter for follower-stopper
    max_decel: float
        maximum deceleration of environment. Used to determine dummy values to put as labels when environment has less vehicles than the maximum amount.

    Returns
    _______
    dict
        Dictionary of trajectory numpy arrays, where matching indeces of each array given (state, action, expert_action, reward, next_state, terminal) tuples
    """

    trajectories = []
    for _ in range(n):

        if multiagent:
            trajectory, length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des)
        else:
            trajectory, length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert, v_des, max_decel)

        trajectories.append((trajectory, length))

    return trajectories


def traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals):
    """
    Collects  observation, action, expert_action, rewards, next observation, terminal lists (collected over a rollout) into a single rollout dictionary.
    Parameters
    __________
    observations: list
        list of observations; ith entry is ith observation
    actions: list
        list of actions; ith entry is action taken at ith timestep
    rewards: list
        list of rewards; ith entry is reward received at ith timestep
    next_observations: list
        list of next observations; ith entry is the observation transitioned to due to state and action at ith timestep
    terminals: list
        list of booleans indicating if rollout ended at that timestep

    Returns
    _______
    dict
        dictionary containing above lists in numpy array form.
    """
    return {"observations" : np.array(observations),
            "actions" : np.array(actions),
            "expert_actions": np.array(expert_actions),
            "rewards" : np.array(rewards),
            "next_observations": np.array(next_observations),
            "terminals": np.array(terminals)}
