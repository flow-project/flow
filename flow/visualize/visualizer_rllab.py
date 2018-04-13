import os
from rllab.sampler.utils import rollout
import argparse
import joblib
import numpy as np
from matplotlib import pyplot as plt
from flow.core.util import emission_to_csv

import pickle

import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of rollouts we will average over')
    parser.add_argument('--plotname', type=str, default="traffic_plot",
                        help='Prefix for all generated plots')
    parser.add_argument('--use_sumogui', action='store_true',
                        help='Flag for using sumo-gui vs sumo binary')
    parser.add_argument('--run_long', type=float, default=1,
                        help='Number by which to increase max_path_length')
    parser.add_argument('--emission_to_csv', action='store_true',
                        help='Specifies whether to convert the emission file '
                             'created by sumo into a csv file')

    args = parser.parse_args()

    data = joblib.load(args.file)
    policy = data['policy']
    baseline = data['baseline']
    env = data['env']

    # FIXME(ev, ak) only one of these should be needed
    # extract the flow environment
    # unwrapped_env = env._wrapped_env._wrapped_env.env.unwrapped
    # if this doesn't work, try the one above it
    unwrapped_env = env.wrapped_env.env.env.unwrapped

    # Input
    if unwrapped_env.obs_var_labels:
        obs_vars = unwrapped_env.obs_var_labels
    else:
        obs_vars = []

    # Recreate experiment params
    vehicles = unwrapped_env.vehicles
    tot_cars = vehicles.num_vehicles
    rl_cars = vehicles.num_rl_vehicles
    max_path_length = int(np.floor(env.horizon*args.run_long))
    flat_obs = env._wrapped_env.observation_space.flat_dim
    num_obs_var = flat_obs / tot_cars

    # TODO: can we do this in a robust way?
    # Recreate the sumo scenario, change the loop length
    scenario = unwrapped_env.scenario
    exp_tag = scenario.name
    net_params = scenario.net_params
    cfg_params = scenario.initial_config
    initial_config = scenario.initial_config

    scenario_class = scenario.__class__
    generator_class = scenario.generator_class

    env._wrapped_env.scenario = scenario

    # update the max path length  # FIXME: still not working
    new_max_path_length = int(np.floor(env.horizon * args.run_long))
    # env_params.additional_params["num_steps"] = new_max_path_length

    # specify an emission path for sumo to generate an emission file for the
    # rollout
    unwrapped_env.emission_path = "./test_time_rollout/"

    # Set sumo to make a video
    sumo_params = unwrapped_env.sumo_params
    sumo_binary = 'sumo-gui' if args.use_sumogui else 'sumo'
    unwrapped_env.restart_sumo(sumo_params, sumo_binary=sumo_binary)

    # Load data into arrays
    all_obs = np.zeros((args.num_rollouts, max_path_length, flat_obs))
    all_rewards = np.zeros((args.num_rollouts, max_path_length))
    for j in range(args.num_rollouts):
        # run a single rollout of the experiment
        path = rollout(env, policy,
                       max_path_length=max_path_length,
                       animated=False, speedup=1)

        # collect the observations and rewards from the rollout
        new_obs = path['observations']
        all_obs[j, :new_obs.shape[0], :new_obs.shape[1]] = new_obs
        new_rewards = path['rewards']
        all_rewards[j, :len(new_rewards)] = new_rewards

        logging.info("\n Done: {0} / {1}, {2}%".format(
            j+1, args.num_rollouts, (j+1) / args.num_rollouts * 100))

    # export observations to a pickle file
    output_filename = 'observations.pkl'
    output = open(output_filename, 'wb')
    pickle.dump(all_obs, output)
    output.close()

    # export rewards to a pickle file
    output_filename = 'rewards.pkl'
    output = open(output_filename, 'wb')
    pickle.dump(all_rewards, output)
    output.close()

    # ensure that a reward_plots folder exists in the directory, and if not,
    # create one
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # create an array of time
    sim_step = sumo_params.sim_step
    t = np.arange(max_path_length) * sim_step

    for obs_var_idx in range(int(num_obs_var)):
        if len(obs_vars) < obs_var_idx + 1:
            obs_var = "Observation {0}".format(obs_var_idx)
        else:
            obs_var = obs_vars[obs_var_idx]

        # plot mean value for observation for each vehicle across rollouts
        plt.figure()
        for car in range(tot_cars):
            center = np.mean(all_obs[:, :, tot_cars*obs_var_idx + car], axis=0)
            plt.plot(range(max_path_length), center, lw=2.0,
                     label='Veh {}'.format(car))
        plt.ylabel(obs_var, fontsize=15)
        plt.xlabel("time (s)", fontsize=15)
        plt.title("{2}, Autonomous Penetration: {0}/{1}".
                  format(rl_cars, tot_cars, obs_var), fontsize=16)
        plt.legend(loc=0)

        # save the plot in the "plots" directory
        plt.savefig("plots/{0}_{1}.png".format(args.plotname, obs_var),
                    bbox="tight")

        # plot mean values for the observations across all vehicles and all
        # rollouts
        car_mean = np.mean(np.mean(
            all_obs[:, :, tot_cars*obs_var_idx:tot_cars*(obs_var_idx + 1)],
            axis=0), axis=1)
        plt.figure()
        plt.plot(t, car_mean)
        plt.ylabel(obs_var, fontsize=15)
        plt.xlabel("time (s)", fontsize=15)
        plt.title("Mean {2}, Autonomous Penetration: {0}/{1}".
                  format(rl_cars, tot_cars, obs_var), fontsize=16)

        # save the plot in the "plots" directory
        plt.savefig("plots/{0}_{1}_mean.png".format(args.plotname, obs_var),
                    bbox="tight")

    # Make a figure for the mean rewards over the course of the rollout
    mean_reward = np.mean(all_rewards, axis=0)

    plt.figure()
    plt.plot(t, mean_reward, lw=2.0)
    plt.ylabel("reward", fontsize=15)
    plt.xlabel("time (s)", fontsize=15)
    plt.title("Reward, Autonomous Penetration: {0}/{1}".
              format(rl_cars, tot_cars), fontsize=16)

    # save the rewards plot in the "reward_plots" directory
    plt.savefig("plots/{0}_reward.png".format(args.plotname), bbox="tight")

    # if prompted, convert the emission file into a csv file
    if args.emission_to_csv:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = "{0}-emission.xml".format(scenario.name)

        emission_path = \
            "{0}/test_time_rollout/{1}".format(dir_path, emission_filename)

        emission_to_csv(emission_path)
