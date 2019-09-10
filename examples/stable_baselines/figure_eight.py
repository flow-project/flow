"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""

import argparse
import json
import os

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params

# time horizon of a single rollout
HORIZON = 1500


# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=13)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='figure_eight_intersection_control',

    # name of the flow environment the experiment is running on
    env_name='AccelEnv',

    # name of the scenario class the experiment is running on
    scenario='FigureEightScenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'sort_vehicles': False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),

)


def run_model(num_cpus=1, rollout_size=50, num_steps=50):
    """Run the model for num_steps if provided. The total rollout length is rollout_size."""
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        env = DummyVecEnv([lambda: constructor])  # The algorithms require a vectorized environment to run
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i) for i in range(num_cpus)])

    model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
    model.learn(total_timesteps=num_steps)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='How many CPUs to use')
    parser.add_argument('--num_steps', type=int, default=5000, help='How many total steps to perform learning over')
    parser.add_argument('--rollout_size', type=int, default=1000, help='How many steps are in a training batch.')
    parser.add_argument('--result_name', type=str, default='figure_eight', help='Name of saved model')
    args = parser.parse_args()
    model = run_model(args.num_cpus, args.rollout_size, args.num_steps)
    # Save the model to a desired folder and then delete it to demonstrate loading
    if not os.path.exists(os.path.realpath(os.path.expanduser('~/baseline_results'))):
        os.makedirs(os.path.realpath(os.path.expanduser('~/baseline_results')))
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    save_path = os.path.join(path, args.result_name)

    print('Saving the trained model!')
    model.save(save_path)
    # dump the flow params
    with open(os.path.join(path, args.result_name) + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,  cls=FlowParamsEncoder, sort_keys=True, indent=4)
    del model
    del flow_params

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    flow_params = get_flow_params(os.path.join(path, args.result_name) + '.json')
    flow_params['sim'].render = True
    env_constructor = env_constructor(params=flow_params, version=0)()
    env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    obs = env.reset()
    reward = 0
    for i in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))
