from rllab.sampler.utils import rollout
import argparse
import joblib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from cistar.scenarios.intersections.intersection_scenario import TwoWayIntersectionScenario
from cistar.scenarios.loop.gen import CircleGenerator
from cistar.scenarios.loop.loop_scenario import LoopScenario
from cistar.scenarios.figure8.figure8_scenario import Figure8Scenario

import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--num_rollouts', type=int, default=100, 
                        help='Number of rollouts we will average over')
    parser.add_argument('--plotname', type=str, default="traffic_plot",
                        help='Prefix for all generated plots')
    parser.add_argument('--use_sumogui', type=bool, default=True,
                        help='Flag for using sumo-gui vs sumo binary')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Number of steps to take')
    parser.add_argument('--loop_length', type=float, default=230,
                        help='Length of loop over which to simulate')
    parser.add_argument('--scenario_type', type=str, default='loop',
                        help='type of scenario being implemented ("loop" or "figure8"')

    args = parser.parse_args()

    # import multiagent.envs as multiagent_envs
    # TODO: distinguish if we have tensorflow active or not
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        # Input
        unwrapped_env = env.env.unwrapped
        if unwrapped_env.obs_var_labels:
            obs_vars = unwrapped_env.obs_var_labels
        else:
            obs_vars = ["Velocity"] # , "Fuel", "Distance"


        # Recreate experiment params
        tot_cars = unwrapped_env.scenario.num_vehicles
        rl_cars = unwrapped_env.scenario.num_rl_vehicles
        # TODO(eugene): fix this to make an observation for each automated
        # vehicle.
        flat_obs = env.observation_space[0].flat_dim
        num_obs_var = flat_obs / tot_cars

        # Recreate the sumo scenario, change the loop length
        scenario = unwrapped_env.scenario
        exp_tag = scenario.name
        type_params = scenario.type_params
        net_params = scenario.net_params
        net_params["length"] = args.loop_length
        cfg_params = scenario.cfg_params
        initial_config = scenario.initial_config
        if args.scenario_type == 'figure8':
            net_params["radius_ring"] = args.radius_ring
            scenario = Figure8Scenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)
        elif args.scenario_type == 'loop':
            net_params["length"] = args.loop_length
            scenario = LoopScenario(exp_tag, CircleGenerator,
                                    type_params, net_params, cfg_params, initial_config=initial_config)
        elif args.scenario_type == 'intersection':
            scenario = TwoWayIntersectionScenario(exp_tag, type_params, net_params, cfg_params, initial_config=initial_config)
        unwrapped_env.scenario = scenario

        # Set sumo to make a video 
        sumo_params = unwrapped_env.sumo_params
        sumo_params['emission_path'] = "./test_time_rollout/"
        sumo_binary = 'sumo-gui' if args.use_sumogui else 'sumo'
        unwrapped_env.restart_sumo(sumo_params, sumo_binary=sumo_binary)


        # Load data into arrays
        all_obs = np.zeros((args.num_rollouts, args.max_path_length, flat_obs))
        all_rewards = np.zeros((args.num_rollouts, args.max_path_length))
        for j in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length,
                       animated=False, speedup=1)
            import ipdb; ipdb.set_trace()
            obs = path['observations'] # length of rollout x flattened observation
            all_obs[j] = obs
            all_rewards[j] = path["rewards"]
            logging.info("\n Done: {0} / {1}, {2}%".format(j+1, args.num_rollouts, (j+1) / args.num_rollouts))
        

        # TODO: savefig doesn't like making new directories
        # Make a separate figure for each observation variable
        for obs_var_idx in range(len(obs_vars)):
            obs_var = obs_vars[obs_var_idx]
            plt.figure()
            for car in range(tot_cars):
                # mean is horizontal, across num_rollouts
                center = np.mean(all_obs[:, :, tot_cars*obs_var_idx + car], axis=0)
                # stdev = np.std(carvels[car], axis=1)
                plt.plot(range(args.max_path_length), center, lw=2.0, label='Car {}'.format(car))
                # plt.plot(range(max_path_length), center + stdev, lw=2.0, c='black', ls=':')
                # plt.plot(range(max_path_length), center - stdev, lw=2.0, c='black', ls=':')
            plt.ylabel(obs_var, fontsize=15)
            plt.xlabel("Rollout/Path Length", fontsize=15)
            plt.title("Cars {0} / {1}".format(rl_cars, tot_cars), fontsize=16)
            plt.legend(loc=0)
            plt.savefig("visualizer/{0}_{1}.png".format(args.plotname, obs_var), bbox="tight")

            # plot mean values across all cars
            car_mean = np.mean(np.mean(all_obs[:, :, tot_cars*obs_var_idx:tot_cars*(obs_var_idx + 1)], 
                            axis=0), axis = 1)
            plt.figure()
            plt.plot(range(args.max_path_length), car_mean)
            plt.ylabel(obs_var, fontsize=15)
            plt.xlabel("Rollout/Path Length", fontsize=15)
            plt.title("Cars {0} / {1}".format(rl_cars, tot_cars), fontsize=16)
            plt.savefig("visualizer/{0}_{1}_mean.png".format(args.plotname, obs_var), bbox="tight")

        # Make a figure for the mean rewards over the course of the rollout
        plt.figure()
        plt.plot(range(args.max_path_length), np.mean(all_rewards, axis=0), lw=2.0)
        plt.ylabel("Reward", fontsize=15)
        plt.xlabel("Rollout/Path Length", fontsize=15)
        plt.title("Cars {0} / {1}".format(rl_cars, tot_cars), fontsize=16)
        plt.savefig("visualizer/{0}_reward.png".format(args.plotname), bbox="tight")
