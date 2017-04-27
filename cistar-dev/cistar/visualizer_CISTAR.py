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

    # currently can't load because env has no scenario

    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    sumo_params = env._wrapped_env.sumo_params
    sumo_params['emission_path'] = "./test_time_rollout/"
    env._wrapped_env.restart_sumo(sumo_params, sumo_binary='sumo-gui')
    # Hacky way to recreate scenario
    tot_cars, auton_cars = 4, 4

    # if env.orig_env.cfgfn[:17] == "/root/code/rllab/":
        # env.orig_env.cfgfn = env.orig_env.cfgfn[17:]
    # Each entry should be a matrix shape length_rollout x num_rollouts
    carvels = [np.zeros((args.max_path_length, args.num_rollouts)) for car in range(tot_cars)]
    for j in range(args.num_rollouts):
        path = rollout(env, policy, max_path_length=args.max_path_length,
                   animated=False, speedup=args.speedup)
        obs = path['observations'] # length of rollout x flattened observation

        for car in range(tot_cars):
            carvels[car][:,j] = obs[:,car]

        print("\n Done: {0} / {1}, {2}%".format(j+1, args.num_rollouts, (j+1) / args.num_rollouts))
    print(obs.shape)
    plt.figure()
    for car in range(tot_cars):
        # mean is horizontal, across num_rollouts
        center = np.mean(carvels[car], axis=1)
        # stdev = np.std(carvels[car], axis=1)
        plt.plot(range(args.max_path_length), center, lw=2.0, label='Car {}'.format(car))
        # plt.plot(range(args.max_path_length), center + stdev, lw=2.0, c='black', ls=':')
        # plt.plot(range(args.max_path_length), center - stdev, lw=2.0, c='black', ls=':')
    plt.ylabel("Velocity m/s", fontsize=15)
    plt.xlabel("Rollout/Path Length", fontsize=15)
    plt.title("Cars {0} / {1} Itr 1000".format(auton_cars, tot_cars), fontsize=16)
    plt.legend(loc=0)
    plt.savefig("visualizer/{0}_velocity.png".format(args.plotname), bbox="tight")

    plt.figure()
    # print(path["rewards"].shape)
    plt.plot(range(args.max_path_length), path["rewards"], lw=2.0)
    plt.ylabel("Reward", fontsize=15)
    plt.xlabel("Rollout/Path Length", fontsize=15)
    plt.title("Cars {0} / {1} Itr 1000".format(auton_cars, tot_cars), fontsize=16)
    plt.savefig("visualizer/{0}_reward.png".format(args.plotname), bbox="tight")
    # print('Total reward: ', sum(path["rewards"]))
    env.terminate()
