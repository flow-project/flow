from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import os
import random
import numpy as np
# import tensorflow as tf
from matplotlib import pyplot as plt

import plotly.offline as po
import plotly.graph_objs as go

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    # parser.add_argument('--max_path_length', type=int, default=1,
    #                     help='Max length of rollout')
    # parser.add_argument('--speedup', type=float, default=1,
    #                     help='Speedup')
    parser.add_argument('--num_rollouts', type=int, default=100, 
                        help='Number of rollouts we will average over')
    parser.add_argument('--plotname', type=str, default="traffic_plot",
                        help='Prefix for all generated plots')
    args = parser.parse_args()

    # Max Path Length should match max path length of original experiment
    # if not max_path_length:
        # raise ValueError("Max Path Length not provided, please match max path length of original experiment")

    # Input
    obs_vars = ["Velocity", "Fuel", "Distance"]

    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    algo = data['algo']

    # Recreate experiment params
    tot_cars = env._wrapped_env.scenario.num_vehicles
    rl_cars = env._wrapped_env.scenario.num_rl_vehicles
    num_itr = algo.n_itr
    max_path_length = algo.max_path_length
    flat_obs = env._wrapped_env.observation_space.flat_dim
    num_obs_var = flat_obs / tot_cars

    # Load data into arrays
    all_obs = np.zeros((args.num_rollouts, max_path_length, flat_obs))
    all_rewards = np.zeros((args.num_rollouts, max_path_length))
    for j in range(args.num_rollouts):
        path = rollout(env, policy, max_path_length=max_path_length,
                   animated=False, speedup=1)
        obs = path['observations'] # length of rollout x flattened observation
        all_obs[j] = obs
        all_rewards[j] = path["rewards"]
        print("\n Done: {0} / {1}, {2}%".format(j+1, args.num_rollouts, (j+1) / args.num_rollouts))
    

    # TODO: savefig doesn't like making new directories
    for obs_var_idx in range(len(obs_vars)):
        obs_var = obs_vars[obs_var_idx]
        plt.figure()
        for car in range(tot_cars):
            # mean is horizontal, across num_rollouts
            center = np.mean(all_obs[:, :, tot_cars*obs_var_idx + car], axis=0)
            # stdev = np.std(carvels[car], axis=1)
            plt.plot(range(max_path_length), center, lw=2.0, label='Car {}'.format(car))
            # plt.plot(range(max_path_length), center + stdev, lw=2.0, c='black', ls=':')
            # plt.plot(range(max_path_length), center - stdev, lw=2.0, c='black', ls=':')
        plt.ylabel(obs_var, fontsize=15)
        plt.xlabel("Rollout/Path Length", fontsize=15)
        plt.title("Cars {0} / {1} Itr {2}".format(rl_cars, tot_cars, num_itr), fontsize=16)
        plt.legend(loc=0)
        plt.savefig("visualizer/{0}_{1}.png".format(args.plotname, obs_var), bbox="tight")

    plt.figure()
    # print(path["rewards"].shape)
    plt.plot(range(max_path_length), np.mean(all_rewards, axis=0), lw=2.0)
    plt.ylabel("Reward", fontsize=15)
    plt.xlabel("Rollout/Path Length", fontsize=15)
    plt.title("Cars {0} / {1} Itr {2}".format(rl_cars, tot_cars, num_itr), fontsize=16)
    plt.savefig("visualizer/{0}_reward.png".format(args.plotname), bbox="tight")
    # print('Total reward: ', sum(path["rewards"]))
    env.terminate()
