"""
Load an already existing Aimsun template and run the simulation
"""

from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.scenarios.loop import Scenario
from flow.controllers.rlcontroller import RLController

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, IDMController, ContinuousRouter


# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2


vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    num_vehicles=2000
)



flow_params = dict(
    # name of the experiment
    exp_tag="test",

    # name of the flow environment the experiment is running on
    env_name="TestEnv",

    # name of the scenario class the experiment is running on
    scenario="Scenario",

    # simulator that is used by the experiment
    simulator='aimsun',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=AimsunParams(
        sim_step=0.1,
        render=True,
        emission_path='data',
    #     scenario_name="Dynamic Scenario 866",
        replication_name="Replication 930",
        centroid_config_name="Centroid Configuration 910"),
    # ),
        # subnetwork_name="Subnetwork 8028981"),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    # net=NetParams(template="/Users/nathan/internship/ring.ang"),
    net=NetParams(template="/Users/nathan/internship/small_test/small_test.ang"),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)



def setup_exps():

    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
        }
    })

