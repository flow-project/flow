"""Purely used to generate files to test rllib functionality"""

import json

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.controllers import IDMController, ContinuousRouter
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS

import os
import unittest

os.environ['TEST_FLAG'] = 'True'

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 2
# number of parallel workers
N_CPUS = 1

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    speed_mode="no_collide",
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="test_rllib",

    # name of the flow environment the experiment is running on
    env_name="TestEnv",

    # name of the scenario class the experiment is running on
    scenario="Figure8Scenario",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=ADDITIONAL_NET_PARAMS
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

class TestVisualizerRLlib(unittest.TestCase):
    """Tests that a clean run of rllib works"""

    def test_rllib(self):
        ray.init(num_cpus=N_CPUS+1, redirect_output=False)

        alg_run = "PPO"

        agent_cls = get_agent_class(alg_run)
        config = agent_cls._default_config.copy()
        config["num_workers"] = N_CPUS

        # save the flow params for replay
        flow_json = json.dumps(
            flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
        config['env_config']['flow_params'] = flow_json
        config['env_config']['run'] = alg_run

        create_env, env_name = make_create_env(params=flow_params, version=0)

        # Register as rllib env
        register_env(env_name, create_env)

        trials = run_experiments({
            flow_params["exp_tag"]: {
                "run": alg_run,
                "env": env_name,
                "config": {
                    **config
                },
                "checkpoint_freq": 1,
                "stop": {
                    "training_iteration": 1
                },
            },
        })
