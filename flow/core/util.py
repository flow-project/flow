"""
A collection of utility functions for Flow

Attributes
----------
E : etree.Element
    Description
"""

import csv
import errno
import importlib
import inspect
import json
import os

from lxml import etree

import xml.etree.ElementTree as ET

from gym.envs.registration import register

from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams

E = etree.Element

class NameEncoder(json.JSONEncoder):

    """
    Custom encoder used to generate ``flow_params.json``
    Extends ``json.JSONEncoder``.
    
    Attributes
    ----------
    allowed_types : list
        A list of primitive types that can be automatically
        serialized by the default JSON encoder
    """

    allowed_types = [dict, list, tuple, str, int, float, bool, type(None)]
    
    def default(self, obj):
        """
        Default encoder (required to extend ``JSONEncoder``)
        
        Parameters
        ----------
        obj : Object
            Object to encode
        
        Returns
        -------
        Object
            A representation of ``obj`` that can be encoded
        """

        if obj not in self.allowed_types:
            if hasattr(obj, '__name__'):
                return obj.__name__
            else:
                return obj.__dict__
        return json.JSONEncoder.default(self, obj)


def unstring_flow_params(flow_params):
    """
    Summary
    
    Parameters
    ----------
    flow_params : dict
        Large dictionary of flow parameters for experiment,
        passed in to ``make_create_env`` and used to create
        ``flow_params.json`` file used by exp visualizer
    
    Returns
    -------
    dict
        Dictionary similar to flow_params but with function
        and module names evaluated, so that the new dict
        can be passed directly to ``make_create_env``
    """

    better_params = flow_params.copy()

    veh_params = flow_params['veh']
    better_veh_params = [eval_veh_params(veh_param) for veh_param in veh_params]
    better_params['veh'] = better_veh_params

    if 'additional_params' in flow_params['env']:
        if 'scenario_type' in flow_params['env']['additional_params']:
            evaluated_scenario = eval(flow_params['env']['additional_params']['scenario_type'])
            better_params['env']['additional_params']['scenario_type'] = evaluated_scenario

    return better_params


def eval_veh_params(orig_params):
    """
    Evaluates vehicle parameters, since params like IDMController can't be
    serialized and output to JSON. Thus, the JSON file stores those as
    their names, with string 'IDMController' instead of the object
    ``<flow.controllers.car_following_models.IDMController``. ``util.py`` 
    imports required car-following models, lane-change controllers, 
    routers, and SUMO parameters. This function evaluates those names
    and returns a dict with the actual objects (evaluated) instead of
    their names.
    
    Parameters
    ----------
    orig_params : dict
        Original vehicle parameters, read in from ``flow_params.json``
    
    Returns
    -------
    dict
        Evaluated vehicle parameters, string names of objects
        replaced with actual objects
    """

    new_params = orig_params.copy()
    
    if 'acceleration_controller' in new_params:
        new_params['acceleration_controller'] = (eval(orig_params['acceleration_controller'][0]), 
                                        orig_params['acceleration_controller'][1])
    if 'lane_change_controller' in new_params:
        new_params['lane_change_controller'] = (eval(orig_params['lane_change_controller'][0]), 
                                        orig_params['lane_change_controller'][1])
    if 'routing_controller' in new_params:
        new_params['routing_controller'] = (eval(orig_params['routing_controller'][0]), 
                                        orig_params['routing_controller'][1])
    if 'sumo_car_following_params' in new_params:
        cf_params = SumoCarFollowingParams()
        cf_params.controller_params = orig_params['sumo_car_following_params']['controller_params']
        new_params['sumo_car_following_params'] = cf_params

    if 'sumo_lc_params' in new_params:
        lc_params = SumoLaneChangeParams()
        lc_params.controller_params = orig_params['sumo_lc_params']['controller_params']
        new_params['sumo_lc_params'] = lc_params

    return new_params


def get_rllib_params(path):
    """
    Returns rllib experiment parameters, given an experiment result folder
    
    Parameters
    ----------
    path : str
        Path to an rllib experiment result directory
    
    Returns
    -------
    dict
        Dictionary of rllib parameters, namely discount factor gamma,
        rollout horizon, and NN hidden layer format
    """

    jsonfile = path + '/params.json'
    jsondata = json.loads(open(jsonfile).read())

    gamma = jsondata['gamma']
    horizon = jsondata['horizon']
    hidden_layers = jsondata['model']['fcnet_hiddens']
    rllib_params = {'gamma':gamma, 
                    'horizon':horizon,
                    'hidden_layers':hidden_layers}
    
    return rllib_params


def get_flow_params(path):
    """
    Returns Flow experiment parameters, given an experiment reuslt folder
    
    Parameters
    ----------
    path : str
        Path to an rllib experiment result directory (``flow_params.json`` is
        in the same folder as ``params.json``)
    
    Returns
    -------
    dict
        Dict of flow parameters, like net_params, env_params, vehicle
        characteristics, etc
    function
        ``make_create_env`` is the higher-order function passed to 
        rllib as the environment in which to train
    """
    
    flow_params_file = path + '/flow_params.json'
    flow_params = json.loads(open(flow_params_file).read())
    flow_params = unstring_flow_params(flow_params)
    
    module_name = 'examples.rllib.' + flow_params['module']

    env_module = importlib.import_module(module_name)
    make_create_env = env_module.make_create_env

    return flow_params, make_create_env


def register_env(env_name, sumo_params, type_params, env_params, net_params,
                initial_config, scenario, env_version_num=0):
    num_steps = 500
    if "num_steps" in env_params.additional_params:
        num_steps = env_params.additional_params["num_steps"]
    register(
        id=env_name+'-v'+str(env_version_num),
        entry_point='flow.envs:'+env_name,
        max_episode_steps=num_steps,
        kwargs={"env_params": env_params, "sumo_params": sumo_params, "scenario": scenario}
    )


def makexml(name, nsl):
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns = {"xsi": xsi}
    attr = {"{%s}noNamespaceSchemaLocation" % xsi: nsl}
    t = E(name, attrib=attr, nsmap=ns)
    return t


def printxml(t, fn):
    etree.ElementTree(t).write(fn, pretty_print=True, encoding='UTF-8',
                               xml_declaration=True)


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def emission_to_csv(emission_path, output_path=None):
    """
    Converts an emission file generated by sumo during an computational
    experiment into a csv file.

    Parameters
    ----------
    emission_path: str
        path to the emission file that should be converted
    output_path: str
        path to the csv file that will be generated, default is the same
        directory as the emission file, with the same name

    Yields
    ------
    A csv file with all the information from the emission file

    Note
    ----
    The emission file contains information generated by sumo, not flow. This
    means tha some data, such as absolute position, is not immediately available
    from the emission file, but can be recreated.
    """
    parser = etree.XMLParser(recover=True)
    tree = ET.parse(emission_path, parser=parser)
    root = tree.getroot()

    # parse the xml data into a dict
    out_data = []
    for time in root.findall('timestep'):
        t = float(time.attrib['time'])

        for car in time:
            out_data.append(dict())
            try:
                out_data[-1]['time'] = t
                out_data[-1]['CO'] = float(car.attrib['CO'])
                out_data[-1]['y'] = float(car.attrib['y'])
                out_data[-1]['CO2'] = float(car.attrib['CO2'])
                out_data[-1]['electricity'] = float(car.attrib['electricity'])
                out_data[-1]['type'] = car.attrib['type']
                out_data[-1]['id'] = car.attrib['id']
                out_data[-1]['eclass'] = car.attrib['eclass']
                out_data[-1]['waiting'] = float(car.attrib['waiting'])
                out_data[-1]['NOx'] = float(car.attrib['NOx'])
                out_data[-1]['fuel'] = float(car.attrib['fuel'])
                out_data[-1]['HC'] = float(car.attrib['HC'])
                out_data[-1]['x'] = float(car.attrib['x'])
                out_data[-1]['route'] = car.attrib['route']
                out_data[-1]['relative_position'] = float(car.attrib['pos'])
                out_data[-1]['noise'] = float(car.attrib['noise'])
                out_data[-1]['angle'] = float(car.attrib['angle'])
                out_data[-1]['PMx'] = float(car.attrib['PMx'])
                out_data[-1]['speed'] = float(car.attrib['speed'])
                out_data[-1]['edge_id'] = car.attrib['lane'].rpartition('_')[0]
                out_data[-1]['lane_number'] = car.attrib['lane'].rpartition('_')[-1]
            except KeyError:
                del out_data[-1]

    # sort the elements of the dictionary by the vehicle id
    out_data = sorted(out_data, key=lambda k: k['id'])

    # default output path
    if output_path is None:
        output_path = emission_path[:-3] + 'csv'

    # output the dict data into a csv file
    keys = out_data[0].keys()
    with open(output_path, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(out_data)
