from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env
from flow.core.params import AimsunParams, NetParams, VehicleParams, EnvParams, InitialConfig
from flow.utils.registry import env_constructor

import os
import json

from coordinated_lights import CoordinatedNetwork, CoordinatedEnv, ADDITIONAL_ENV_PARAMS

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

sim_step = 0.8 # seconds
detector_step = 60  # seconds
timehorizon = 3600
HORIZON = int(timehorizon//sim_step)

net_params = NetParams(template=os.path.abspath("no_api_scenario.ang"))
initial_config = InitialConfig()
vehicles = VehicleParams()
env_params = EnvParams(horizon=HORIZON,
                       sims_per_step=int(detector_step/sim_step),
                       additional_params=ADDITIONAL_ENV_PARAMS)
sim_params = AimsunParams(sim_step=sim_step,
                          render=False,
                          restart_instance=False,
                          replication_name="Replication (one hour)",
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
    num_cpus = 2
    num_rollouts = 8
    rollout_size = int(timehorizon/detector_step)
    num_steps = int(timehorizon/detector_step)*num_rollouts
    result_name = "result_demo"
    model = run_model(num_cpus, rollout_size, num_steps)
    # Save the model to a desired folder and then delete it to demonstrate loading
    if not os.path.exists(os.path.realpath(os.path.expanduser('~/baseline_results'))):
        os.makedirs(os.path.realpath(os.path.expanduser('~/baseline_results')))
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    save_path = os.path.join(path, result_name)
    print('Saving the trained model!')
    model.save(save_path)
