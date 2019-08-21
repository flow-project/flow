"""Open merge example.

Trains a a small percentage of rl vehicles to dissipate shockwaves caused by
merges in an open network.
"""
import argparse
import json
import os

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from flow.core.params import SumoParams, EnvParams, InitialConfig, InFlows, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params

# experiment number
# - 0: 10% RL penetration,  5 max controllable vehicles
# - 1: 25% RL penetration, 13 max controllable vehicles
# - 2: 33% RL penetration, 17 max controllable vehicles
EXP_NUM = 0

# time horizon of a single rollout
HORIZON = 600
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

# inflow rate at the highway
FLOW_RATE = 2000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [5, 13, 17][EXP_NUM]

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 500

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=100,
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="stabilizing_open_network_merges",

    # name of the flow environment the experiment is running on
    env_name="MergePOEnv",

    # name of the scenario class the experiment is running on
    scenario="MergeScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=5,
        warmup_steps=0,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "num_rl": NUM_RL,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
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
    parser.add_argument('--result_name', type=str, default='stabilize_highway', help='Name of saved model')
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
        json.dump(flow_params, outfile, cls=FlowParamsEncoder, sort_keys=True, indent=4)
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
