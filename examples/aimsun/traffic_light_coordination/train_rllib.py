from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env
from flow.core.params import AimsunParams, NetParams, VehicleParams, EnvParams, InitialConfig

import os
import ray
import json

from coordinated_lights import CoordinatedNetwork, CoordinatedEnv, ADDITIONAL_ENV_PARAMS

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


sim_step = 0.8  # seconds
# detector_step = 300  # seconds
detector_step = 60  # seconds

timehorizon = 3600*4 - detector_step
HORIZON = int(timehorizon//sim_step)
N_ROLLOUTS = 1

net_params = NetParams(template=os.path.abspath("no_api_scenario.ang"))
initial_config = InitialConfig()
vehicles = VehicleParams()
env_params = EnvParams(horizon=HORIZON,
                       warmup_steps=int(detector_step/sim_step),
                       sims_per_step=int(detector_step/sim_step),
                       additional_params=ADDITIONAL_ENV_PARAMS)
sim_params = AimsunParams(sim_step=sim_step,
                          render=False,
                          replication_name="Replication 8050297",
                          centroid_config_name="Centroid Configuration 8040652"
                          )


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
    config["num_workers"] = N_CPUS
    # config["sgd_minibatch_size"] = 120  # remove me
    config["train_batch_size"] = HORIZON*N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON
    config["vf_loss_coeff"] = 1e-5

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=version)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    N_CPUS = 2
    ray.init(num_cpus=N_CPUS + 1, object_store_memory=int(1e9))

    alg_run, gym_name, config = setup_exps()
    trials = run_experiments({
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
                "training_iteration": 1,
            },
            "resume": True,
            # "local_dir": os.path.abspath("./ray_results"),
        }
    })
