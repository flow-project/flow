"""Utility method for registering environments with OpenAI gym."""

import gym
from gym.envs.registration import register

from copy import deepcopy

import flow.envs
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams


def make_create_env(params, version=0, render=None):
    """Create a parametrized flow environment compatible with OpenAI gym.

    This environment creation method allows for the specification of several
    key parameters when creating any flow environment, including the requested
    environment and network classes, and the inputs needed to make these
    classes generalizable to networks of varying sizes and shapes, and well as
    varying forms of control (e.g. AVs, automated traffic lights, etc...).

    This method can also be used to recreate the environment a policy was
    trained on and assess it performance, or a modified form of the previous
    environment may be used to profile the performance of the policy on other
    types of networks.

    Parameters
    ----------
    params : dict
        flow-related parameters, consisting of the following keys:

         - exp_tag: name of the experiment
         - env_name: environment class of the flow environment the experiment
           is running on. (note: must be in an importable module.)
         - network: network class the experiment uses.
         - simulator: simulator that is used by the experiment (e.g. aimsun)
         - sim: simulation-related parameters (see flow.core.params.SimParams)
         - env: environment related parameters (see flow.core.params.EnvParams)
         - net: network-related parameters (see flow.core.params.NetParams and
           the network's documentation or ADDITIONAL_NET_PARAMS component)
         - veh: vehicles to be placed in the network at the start of a rollout
           (see flow.core.params.VehicleParams)
         - initial (optional): parameters affecting the positioning of vehicles
           upon initialization/reset (see flow.core.params.InitialConfig)
         - tls (optional): traffic lights to be introduced to specific nodes
           (see flow.core.params.TrafficLightParams)

    version : int, optional
        environment version number
    render : bool, optional
        specifies whether to use the gui during execution. This overrides
        the render attribute in SumoParams

    Returns
    -------
    function
        method that calls OpenAI gym's register method and make method
    str
        name of the created gym environment
    """
    exp_tag = params["exp_tag"]

    if isinstance(params["env_name"], str):
        print("""Passing of strings for env_name will be deprecated.
        Please pass the Env instance instead.""")
        base_env_name = params["env_name"]
    else:
        base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)

    if isinstance(params["network"], str):
        print("""Passing of strings for network will be deprecated.
        Please pass the Network instance instead.""")
        module = __import__("flow.networks", fromlist=[params["network"]])
        network_class = getattr(module, params["network"])
    else:
        network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        network = network_class(
            name=exp_tag,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        # accept new render type if not set to None
        sim_params.render = render or sim_params.render

        # check if the environment is a single or multiagent environment, and
        # get the right address accordingly
        single_agent_envs = [env for env in dir(flow.envs)
                             if not env.startswith('__')]

        if isinstance(params["env_name"], str):
            if params['env_name'] in single_agent_envs:
                env_loc = 'flow.envs'
            else:
                env_loc = 'flow.envs.multiagent'
            entry_point = env_loc + ':{}'.format(params["env_name"])
        else:
            entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

        # register the environment with OpenAI gym
        register(
            id=env_name,
            entry_point=entry_point,
            kwargs={
                "env_params": env_params,
                "sim_params": sim_params,
                "network": network,
                "simulator": params['simulator']
            })

        return gym.envs.make(env_name)

    return create_env, env_name


def env_constructor(params, version=0, render=None):
    """Return a constructor from make_create_env."""
    create_env, env_name = make_create_env(params, version, render)
    return create_env
