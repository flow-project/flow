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

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=400,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--num_rollouts', type=int, default=100, 
                        help='Number of rollouts we will average over')
    parser.add_argument('--plotname', type=str, default="traffic_plot",
                        help='Prefix for all generated plots')
    args = parser.parse_args()

    # policy = None
    # env = None

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    # while True:
    # with tf.Session() as sess:
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    env.orig_env.initialized = False
    if env.orig_env.cfgfn[:17] == "/root/code/rllab/":
        env.orig_env.cfgfn = env.orig_env.cfgfn[17:]
    plt.figure()
    # Each entry should be a matrix shape length_rollout x num_rollouts
    carvels = [np.zeros((args.max_path_length, args.num_rollouts)) for car in range(env.orig_env.num_cars)]
    for j in range(args.num_rollouts):
        path = rollout(env, policy, max_path_length=args.max_path_length,
                   animated=False, speedup=args.speedup)
        obs = path['observations'] # length of rollout x flattened observation
        for car in range(env.orig_env.num_cars):
            carvels[car][:,j] = obs[:,car*3]
    # print("Car velocities shapes")
    for car in range(env.orig_env.num_cars):
        # print(car, carvels[car].shape)
        # mean is horizontal, across num_rollouts
        center = np.mean(carvels[car], axis=1)
        stdev = np.std(carvels[car], axis=1)
        plt.plot(range(args.max_path_length), center, lw=2.0, label='Car {}'.format(car))
        plt.plot(range(args.max_path_length), center + stdev, lw=2.0, c='black', ls=':')
        plt.plot(range(args.max_path_length), center - stdev, lw=2.0, c='black', ls=':')
    plt.ylabel("Velocity m/s", fontsize=15)
    plt.xlabel("Rollout/Path Length", fontsize=15)
    plt.title("Cars 2 / 12 Itr 250", fontsize=16)
    plt.legend(loc=0)
    plt.savefig("{0}_velocity.png".format(args.plotname), bbox="tight")
    
    plt.figure()
    print(path["rewards"].shape)
    plt.plot(range(args.max_path_length), path["rewards"], lw=2.0)
    plt.ylabel("Reward", fontsize=15)
    plt.xlabel("Rollout/Path Length", fontsize=15)
    plt.title("Cars 2 / 12 Itr 250", fontsize=16)
    plt.savefig("{0}_reward.png".format(args.plotname), bbox="tight")
    # print('Total reward: ', sum(path["rewards"]))
    env.orig_env.close()
        # args.loop -= 1
        # if ':' not in args.file:
        #     if args.loop <= 0:
        #         break
        #     while True:
        #         path = rollout(env, policy, max_path_length=args.max_path_length,
        #                        animated=True, speedup=args.speedup)
