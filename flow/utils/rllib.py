"""
Utility functions for Flow compatibility with RLlib, including: environment
generation, serialization, and visualization.
"""
import json

import gym
from gym.envs.registration import register

from flow.core.params import SumoLaneChangeParams, SumoCarFollowingParams, \
    SumoParams, InitialConfig, EnvParams, NetParams, InFlows
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles


def make_create_env(params, version=0, sumo_binary=None):
    """Creates a parametrized flow environment compatible with RLlib.

    This environment creation method allows for the specification of several
    key parameters when creating any flow environment, including the requested
    environment, scenario, and generator classes, and the inputs needed to make
    these classes generalizable to networks of varying sizes and shapes, and
    well as varying forms of control (e.g. AVs, automated traffic lights,
    etc...).

    This method can also be used to recreate the environment a policy was
    trained on and asses it performance, or a modified form of the previous
    environment may be used to profile the performance of the policy on other
    types of networks.

    Parameters
    ----------
    params : dict
        flow-related parameters, consisting of the following keys:
         - exp_tag: name of the experiment
         - env_name: name of the flow environment the experiment is running on
         - scenario: name of the scenario class the experiment uses
         - generator: name of the generator used to create/modify the network
           configuration files
         - sumo: sumo-related parameters (see flow.core.params.SumoParams)
         - env: environment related parameters (see flow.core.params.EnvParams)
         - net: #network-related parameters (see flow.core.params.NetParams and
           the scenario's documentation or ADDITIONAL_NET_PARAMS component)
         - veh: vehicles to be placed in the network at the start of a rollout
           (see flow.core.vehicles.Vehicles)
         - initial (optional): parameters affecting the positioning of vehicles
           upon initialization/reset (see flow.core.params.InitialConfig)
         - tls (optional): traffic lights to be introduced to specific nodes
           (see flow.core.traffic_lights.TrafficLights)
    version : int, optional
        environment version number
    sumo_binary : bool, optional
        specifies whether to use sumo's gui during execution. This overrides
        the sumo_binary component in SumoParams

    Returns
    -------
    function
        method that calls OpenAI gym's register method, and is parametrized by
        an env_config component which is used by RLlib
    str
        name of the created gym environment

    """
    exp_tag = params["exp_tag"]

    env_name = params["env_name"] + '-v{}'.format(version)

    module = __import__("flow.scenarios", fromlist=[params["scenario"]])
    scenario_class = getattr(module, params["scenario"])
    module = __import__("flow.scenarios", fromlist=[params["generator"]])
    generator_class = getattr(module, params["generator"])

    sumo_params = params['sumo']
    env_params = params['env']
    net_params = params['net']
    vehicles = params['veh']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLights())

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    scenario = scenario_class(
        name=exp_tag,
        generator_class=generator_class,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    def create_env(*_):
        register(
            id=env_name,
            entry_point='flow.envs:' + params["env_name"],
            max_episode_steps=env_params.horizon,
            kwargs={
                "env_params": env_params,
                "sumo_params": sumo_params,
                "scenario": scenario
            }
        )
        return gym.envs.make(env_name)

    return create_env, env_name


class FlowParamsEncoder(json.JSONEncoder):
    """
    Custom encoder used to generate ``flow_params.json``
    Extends ``json.JSONEncoder``.
    """

    def default(self, obj):
        allowed_types = [dict, list, tuple, str, int, float, bool, type(None)]

        if obj not in allowed_types:
            if isinstance(obj, Vehicles):
                res = obj.initial
                for res_i in res:
                    res_i["acceleration_controller"] = \
                        (res_i["acceleration_controller"][0].__name__,
                         res_i["acceleration_controller"][1])
                    res_i["lane_change_controller"] = \
                        (res_i["lane_change_controller"][0].__name__,
                         res_i["lane_change_controller"][1])
                    if res_i["routing_controller"] is not None:
                        res_i["routing_controller"] = \
                            (res_i["routing_controller"][0].__name__,
                             res_i["routing_controller"][1])
                return obj.initial
            if hasattr(obj, '__name__'):
                return obj.__name__
            else:
                return obj.__dict__

        return json.JSONEncoder.default(self, obj)


def get_flow_params(config):
    """Returns Flow experiment parameters, given an experiment result folder

    Parameters
    ----------
    config : dict
        stored RLlib configuration dict

    Returns
    -------
    dict
        Dict of flow parameters, like net_params, env_params, vehicle
        characteristics, etc
    """
    # collect all data from the json file
    flow_params = json.loads(config['env_config']['flow_params'])

    # reinitialize the vehicles class from stored data
    veh = Vehicles()
    for veh_params in flow_params["veh"]:
        module = __import__(
            "flow.controllers",
            fromlist=[veh_params['acceleration_controller'][0]]
        )
        acc_class = getattr(module, veh_params['acceleration_controller'][0])
        lc_class = getattr(module, veh_params['lane_change_controller'][0])

        acc_controller = (acc_class, veh_params['acceleration_controller'][1])
        lc_controller = (lc_class, veh_params['lane_change_controller'][1])

        rt_controller = None
        if veh_params['routing_controller'] is not None:
            rt_class = getattr(module, veh_params['routing_controller'][0])
            rt_controller = (rt_class, veh_params['routing_controller'][1])

        sumo_cf_params = SumoCarFollowingParams()
        sumo_cf_params.__dict__ = veh_params["sumo_car_following_params"]

        sumo_lc_params = SumoLaneChangeParams()
        sumo_lc_params.__dict__ = veh_params["sumo_lc_params"]

        del veh_params["sumo_car_following_params"], \
            veh_params["sumo_lc_params"], \
            veh_params["acceleration_controller"], \
            veh_params["lane_change_controller"], \
            veh_params["routing_controller"]

        veh.add(acceleration_controller=acc_controller,
                lane_change_controller=lc_controller,
                routing_controller=rt_controller,
                sumo_car_following_params=sumo_cf_params,
                sumo_lc_params=sumo_lc_params,
                **veh_params)

    # convert all parameters from dict to their object form
    sumo = SumoParams()
    sumo.__dict__ = flow_params["sumo"].copy()

    net = NetParams()
    net.__dict__ = flow_params["net"].copy()
    net.in_flows = InFlows()
    if flow_params["net"]["in_flows"]:
        net.in_flows.__dict__ = flow_params["net"]["in_flows"].copy()

    env = EnvParams()
    env.__dict__ = flow_params["env"].copy()

    initial = InitialConfig()
    if "initial" in flow_params:
        initial.__dict__ = flow_params["initial"].copy()

    tls = TrafficLights()
    if "tls" in flow_params:
        initial.__dict__ = flow_params["tls"].copy()

    flow_params["sumo"] = sumo
    flow_params["env"] = env
    flow_params["initial"] = initial
    flow_params["net"] = net
    flow_params["veh"] = veh
    flow_params["tls"] = tls

    return flow_params
