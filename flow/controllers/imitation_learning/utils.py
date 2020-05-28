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
                        print("Controller collecing trajectory: ", type(expert))
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

    observation_dict = env.reset()

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:


        # vehicle_ids = env.k.vehicle.get_rl_ids() **this doesn't work now due to control range restriction**
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
    Samples trajectories to collect at least min_batch_timesteps steps in the environment

    Args:
        env: environment
        vehicle_id: id of vehicle being tracked/controlled
        controller: subclass of BaseController, decides actions taken by vehicle
        expert_controller: subclass of BaseController, "expert" for imitation learning
        min_batch_timesteps: minimum number of environment steps to collect
        max_trajectory_length: maximum steps in a trajectory
        v_des: parameter used for follower-stopper (applies if Expert controller is follower-stopper)

    Returns:
        List of rollout dictionaries, total steps taken by environment
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
    Collects a fixed number of trajectories.

    Args:
        env: environment
        vehicle_id: id of vehicle being tracked/controlled
        controller: subclass of BaseController, decides actions taken by vehicle
        expert_controller: subclass of BaseController, "expert" for imitation learning
        n: number of trajectories to collect
        max_trajectory_length: maximum steps in a trajectory
        v_des: parameter used for follower-stopper (applies if Expert controller is follower-stopper)


    Returns:
        List of rollouts (tuple of rollout dictionary, length of rollout)

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
    Collects individual observation, action, expert_action, rewards, next observation, terminal arrays into a single rollout dictionary
    """
    return {"observations" : np.array(observations),
            "actions" : np.array(actions),
            "expert_actions": np.array(expert_actions),
            "rewards" : np.array(rewards),
            "next_observations": np.array(next_observations),
            "terminals": np.array(terminals)}
