import tensorflow as tf
import os
import numpy as np
import math
from flow.core.params import SumoCarFollowingParams
from imitating_controller import ImitatingController
from imitating_network import ImitatingNetwork
from flow.controllers.car_following_models import IDMController
from flow.controllers.velocity_controllers import FollowerStopper

""" Class agnostic helper functions """

def sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert):
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

    vehicle_ids = env.k.vehicle.get_rl_ids()
    print("VEHICLE IDS: ", vehicle_ids)
    assert len(vehicle_ids) <= 1, "Not single-agent"
    observation = env.reset()

    if len(vehicle_ids) == 1:
        vehicle_id = vehicle_ids[0]
    else:
        vehicle_id = None

    observations, actions, expert_actions, rewards, next_observations, terminals = [], [], [], [], [], []
    traj_length = 0

    while True:

        # update vehicle ids and make sure it is single agent
        vehicle_ids = env.k.vehicle.get_rl_ids()
        if len(vehicle_ids) == 0:
            observation, reward, done, _ = env.step(None)
            if done:
                break
            continue

        assert len(vehicle_ids) == 1, "Not single agent"

        # init controllers if vehicle id is new
        vehicle_id = vehicle_ids[0]
        if vehicle_id not in set(controllers.get_keys()):

            expert = FollowerStopper(vehicle_id, car_following_params=car_following_params)
            imitator = ImitatingController(vehicle_id, action_network, false, car_following_params=car_following_params)
            controllers[vehicle_id] = (imitator, expert)

        # decide which controller to use to collect trajectory
        expert_controller = controllers[vehicle_id][1]
        if use_expert:
            controller = expert_controller
        else:
            controller = controllers[vehicle_id][0]


        print("COLLECTING CONTROLLER: ", controller)
        print("EXPERT CONTROLLER: ", expert_controller)

        action = controller.get_action(env)
        if type(action) == np.ndarray:
            action = action.flatten()[0]

        expert_action = expert_controller.get_action(env)
        if (expert_action is None or math.isnan(expert_action)):
            observation, reward, done, _ = env.step(action)
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

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals), traj_length


def sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert):
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
        vehicle_ids = env.k.vehicle.get_rl_ids()
        if len(vehicle_ids) == 0:
            print("NO RL VEHICLES")
            observation_dict, reward, done, _ = env.step(None)
            print(env.k.vehicle.get_rl_ids())
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

                expert = FollowerStopper(vehicle_id, car_following_params=car_following_params)
                imitator = ImitatingController(vehicle_id, action_network, True, car_following_params=car_following_params)
                controllers[vehicle_id] = (imitator, expert)

            expert_controller = controllers[vehicle_id][1]
            if use_expert:
                controller = expert_controller
            else:
                controller = controllers[vehicle_id][0]

            if traj_length == 0 and i == 0:
                print("COLLECTOR: ", controller)

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
            next_observations.append(observation_dict[vehicle_id])
            rewards.append(reward_dict[vehicle_id])
            terminals.append(terminate_rollout)

        traj_length += 1

        if terminate_rollout:
            break

    return traj_dict(observations, actions, expert_actions, rewards, next_observations, terminals), traj_length


def sample_trajectories(env, controllers, action_network, min_batch_timesteps, max_trajectory_length, multiagent, use_expert):
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
            trajectory, traj_length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert)
        else:
            trajectory, traj_length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert)

        trajectories.append(trajectory)

        total_envsteps += traj_length

    return trajectories, total_envsteps

def sample_n_trajectories(env, controllers, action_network, n, max_trajectory_length, multiagent, use_expert):
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
            trajectory, length = sample_trajectory_multiagent(env, controllers, action_network, max_trajectory_length, use_expert)
        else:
            trajectory, length = sample_trajectory_singleagent(env, controllers, action_network, max_trajectory_length, use_expert)

        trajectories.append((trajectory, length))

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
