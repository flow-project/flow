import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


# Intersection params for two way intersection
def pass_params(env_name, sumo_params, sumo_binary,
                type_params, env_params, net_params,
                cfg_params, initial_config, scenario):
    num_steps = 500
    if "num_steps" in env_params.additional_params:
        num_steps = env_params.get_additional_param["num_steps"]
    register(
        id=env_name+'-v0',
        entry_point='flow.envs:'+env_name,
        max_episode_steps=num_steps,
        kwargs={"env_params": env_params, "sumo_binary": sumo_binary,
                "sumo_params": sumo_params, "scenario": scenario}
    )
