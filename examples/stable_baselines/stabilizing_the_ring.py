"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""

import argparse
import os

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from flow.utils.registry import construct_env
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter

# time horizon of a single rollout
HORIZON = 3000

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_the_ring",

    # name of the flow environment the experiment is running on
    env_name="WaveAttenuationPOEnv",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [220, 270],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='How many CPUs to use')
    parser.add_argument('--num_steps', type=int, default=5000, help='How many total steps to perform learning over')
    parser.add_argument('--rollout_size', type=int, default=1000, help='How many steps are in a training batch.')
    parser.add_argument('--result_name', type=str, default='ring_stabilize', help='Name of saved model')
    args = parser.parse_args()

    if args.num_cpus == 1:
        env_constructor = construct_env(params=flow_params, version=0)()
        env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    else:
        env = SubprocVecEnv([construct_env(params=flow_params, version=i) for i in range(args.num_cpus)])

    model = PPO2('MlpPolicy', env, verbose=1, n_steps=args.rollout_size)
    model.learn(total_timesteps=args.num_steps)

    # Save the model to a desired folder and then delete it to demonstrate loading
    if not os.path.exists(os.path.realpath(os.path.expanduser('~/baseline_results'))):
        os.makedirs(os.path.realpath(os.path.expanduser('~/baseline_results')))
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    save_path = os.path.join(path, args.result_name)
    model.save(save_path)
    del model

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    flow_params['sim'].render = True
    env_constructor = construct_env(params=flow_params, version=0)()
    env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    obs = env.reset()
    reward = 0
    for i in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))
