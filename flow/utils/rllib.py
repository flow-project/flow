"""
Utility functions for Flow compatibility with RLlib.

This includes: environment generation, serialization, and visualization.
"""
import dill
import json
from copy import deepcopy

from flow.core.params import SumoLaneChangeParams, SumoCarFollowingParams, \
    SumoParams, InitialConfig, EnvParams, NetParams, InFlows
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles


class FlowParamsEncoder(json.JSONEncoder):
    """
    Custom encoder used to generate ``flow_params.json``.

    Extends ``json.JSONEncoder``.
    """

    def default(self, obj):
        """See parent class.

        Extended to support the Vehicles object in flow/core/vehicles.py.
        """
        allowed_types = [dict, list, tuple, str, int, float, bool, type(None)]

        if obj not in allowed_types:
            if isinstance(obj, Vehicles):
                res = deepcopy(obj.initial)
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
                return res
            if hasattr(obj, '__name__'):
                return obj.__name__
            else:
                return obj.__dict__

        return json.JSONEncoder.default(self, obj)


def get_flow_params(config):
    """Return Flow experiment parameters, given an experiment result folder.

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
            fromlist=[veh_params['acceleration_controller'][0]])
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

        veh.add(
            acceleration_controller=acc_controller,
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
    net.inflows = InFlows()
    if flow_params["net"]["inflows"]:
        net.inflows.__dict__ = flow_params["net"]["inflows"].copy()

    env = EnvParams()
    env.__dict__ = flow_params["env"].copy()

    initial = InitialConfig()
    if "initial" in flow_params:
        initial.__dict__ = flow_params["initial"].copy()

    tls = TrafficLights()
    if "tls" in flow_params:
        tls.__dict__ = flow_params["tls"].copy()

    flow_params["sumo"] = sumo
    flow_params["env"] = env
    flow_params["initial"] = initial
    flow_params["net"] = net
    flow_params["veh"] = veh
    flow_params["tls"] = tls

    return flow_params


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + '/params.json'  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + '/params.pkl'  # params.json is the config file
    with open(pklfile, 'rb') as file:
        pkldata = dill.load(file)
    return pkldata
