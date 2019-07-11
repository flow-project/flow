"""Trains vehicles to facilitate cooperative merging in a loop merge.

This examples consists of 1 learning agent and 6 additional vehicles in an
inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring.
"""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.controllers import RLController
from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.controllers import SimLaneChangeController
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import VehicleParams

# time horizon of a single rollout
HORIZON = 100
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 2

RING_RADIUS = 100
NUM_MERGE_HUMANS = 9
NUM_MERGE_RL = 1

# note that the vehicles are added sequentially by the scenario,
# so place the merging vehicles after the vehicles in the ring
vehicles = VehicleParams()
# Inner ring vehicles
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=6,
    car_following_params=SumoCarFollowingParams(minGap=0.0, tau=0.5),
    lane_change_params=SumoLaneChangeParams())
# A single learning agent in the inner ring
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1,
    car_following_params=SumoCarFollowingParams(
        minGap=0.01,
        tau=0.5,
        speed_mode="obey_safe_speed",
    ),
    lane_change_params=SumoLaneChangeParams())
# Outer ring vehicles
vehicles.add(
    veh_id='merge-human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=10,
    car_following_params=SumoCarFollowingParams(minGap=0.0, tau=0.5),
    lane_change_params=SumoLaneChangeParams())

flow_params = dict(
    # name of the experiment
    exp_tag='cooperative_merge',

    # name of the flow environment the experiment is running on
    env_name='AccelEnv',

    # name of the scenario class the experiment is running on
    scenario='TwoLoopsOneMergingScenario',

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
            "target_velocity": 10,
            "max_accel": 3,
            "max_decel": 3,
            "sort_vehicles": False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params={
            'ring_radius': 50,
            'lane_length': 75,
            'inner_lanes': 1,
            'outer_lanes': 1,
            'speed_limit': 30,
            'resolution': 40,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        x0=50,
        spacing='uniform',
        additional_params={
            'merge_bunching': 0,
        },
    ),
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [16, 16, 16]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['horizon'] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS+1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 20,
            "checkpoint_at_end": True,
            'max_failures': 999,
            'stop': {
                'training_iteration': 200,
            },
        }
    })
