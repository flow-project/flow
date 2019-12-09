import os
import json

import ray
import numpy as np

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env
from flow.core.params import AimsunParams, NetParams, VehicleParams, EnvParams, InitialConfig

from coordinated_lights import CoordinatedNetwork, CoordinatedEnv, ADDITIONAL_ENV_PARAMS

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


SIM_STEP = 0.8  # copy to run.py

# hardcoded to AIMSUN's statistics update interval (5 minutes)
DETECTOR_STEP = 300  # copy to run.py

TIME_HORIZON = 3600*4 - DETECTOR_STEP  # 14,100
HORIZON = int(TIME_HORIZON//SIM_STEP)

RLLIB_N_CPUS = 2
RLLIB_HORIZON = int(TIME_HORIZON//DETECTOR_STEP)  # 47

RLLIB_N_ROLLOUTS = 3  # copy to coordinated_lights.py
RLLIB_TRAINING_ITERATIONS = 1000

net_params = NetParams(template=os.path.abspath("no_api_scenario.ang"))
initial_config = InitialConfig()
vehicles = VehicleParams()
env_params = EnvParams(horizon=HORIZON,
                       warmup_steps=int(np.ceil(120/DETECTOR_STEP)),
                       sims_per_step=int(DETECTOR_STEP/SIM_STEP),
                       additional_params=ADDITIONAL_ENV_PARAMS)
sim_params = AimsunParams(sim_step=SIM_STEP,
                          render=False,
                          restart_instance=False,
                          #   replication_name="Replication (one hour)",
                          replication_name=ADDITIONAL_ENV_PARAMS['replication_list'][0],
                          centroid_config_name="Centroid Configuration 8040652")


flow_params = dict(
    exp_tag="coordinated_traffic_lights",
    env_name=CoordinatedEnv,
    network=CoordinatedNetwork,
    simulator='aimsun',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)


def setup_exps(version=0):
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
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = RLLIB_N_CPUS
    # config["sgd_minibatch_size"] = 32  # remove me
    config["train_batch_size"] = RLLIB_HORIZON * RLLIB_N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["sample_batch_size"] = RLLIB_HORIZON * RLLIB_N_ROLLOUTS
    config["model"].update({"fcnet_hiddens": [64, 64, 64]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # (ev) temporary ray bug
    config["horizon"] = RLLIB_HORIZON  # not same as env horizon.
    config["vf_loss_coeff"] = 1e-8
    config["vf_clip_param"] = 600
    config["lr"] = 5e-4

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=version)

    # Register as rllib env
    ray.tune.registry.register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    ray.init(num_cpus=RLLIB_N_CPUS + 1, object_store_memory=int(1e9))

    alg_run, gym_name, config = setup_exps()
    trials = ray.tune.run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": RLLIB_TRAINING_ITERATIONS,
            },
            # "restore": '/Users/umang/ray_results/coordinated_traffic_lights/PPO_CoordinatedEnv-v0_0_2019-12-07_16-03-40qh2emtjb/checkpoint_48/checkpoint-48'
            # "local_dir": os.path.abspath("./ray_results"),
        }
    }, resume=False)
