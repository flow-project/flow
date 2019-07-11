# flake8: noqa
"""Script to load an Aimsun instance from a template."""
import os
import json

import flow.config as config
from flow.utils.aimsun.scripting_api import AimsunTemplate


def load_network():
    """Load the whole network into a dictionary and returns it."""
    sections = model.sections
    nodes = model.nodes
    turnings = model.turnings
    cen_connections = model.cen_connections

    scenario_data = get_dict_from_objects(sections, nodes, turnings,
                                          cen_connections)
    return scenario_data


def load_subnetwork(subnetwork, scenario):
    """Load subnetwork into a dictionary and returns it."""
    # get all objects in subnetwork
    objs = list(subnetwork.classify_objects(scenario.id))

    sections = model.find_all_by_type(objs, 'GKSection')
    nodes = model.find_all_by_type(objs, 'GKNode')
    turnings = model.find_all_by_type(objs, 'GKTurning')
    cen_connections = model.find_all_by_type(objs, 'GKCenConnection')

    scenario_data = get_dict_from_objects(sections, nodes, turnings,
                                          cen_connections)
    return scenario_data


def get_dict_from_objects(sections, nodes, turnings, cen_connections):
    """Load all relevant data into dictionaries."""
    scenario_data = {
        'sections': {},
        'nodes': {},
        'turnings': {},
        'centroids': {},
        'connections': {}
    }

    # load centroids
    # TODO use centroids when we don't have a centroid configuration
    # centroids = [o for o in objs if o.getTypeName() == 'GKCentroid']
    # FIXME doesn't handle centroids that are both in and out
    # maybe no need to distinguish them since it is done
    # later with centroid connections
    centroid_config_name = data['centroid_config_name']
    centroid_config = model.find_by_name(model.centroid_configurations,
                                         centroid_config_name)
    if not centroid_config:
        print('[load.py] ERROR: Centroid configuration ' +
              centroid_config_name + ' does not exist.')
    for c in centroid_config.origin_centroids:
        scenario_data['centroids'][c.id] = {'type': 'in'}
    for c in centroid_config.destination_centroids:
        scenario_data['centroids'][c.id] = {'type': 'out'}

    # load sections
    for s in sections:
        scenario_data['sections'][s.id] = {
            'name': s.name,
            'numLanes': s.nb_full_lanes,
            # FIXME this is a mean of the lanes lengths
            # (bc they don't have to be all of the same size)
            # it may not be 100% accurate
            'length': s.lanes_length_2D / s.nb_full_lanes,
            'max_speed': s.speed
        }

    # load nodes
    for n in nodes:
        scenario_data['nodes'][n.id] = {
            'name': n.name,
            'nb_turnings': len(n.turnings)
        }

    # load turnings
    for t in turnings:
        scenario_data['turnings'][t.id] = {
            'name': t.name,
            'length': t.polygon.length2D() / 2,  # FIXME not totally accurate
            'origin_section_name': t.origin.name,
            'origin_section_id': t.origin.id,
            'dest_section_name': t.destination.name,
            'dest_section_id': t.destination.id,
            'node_id': t.node.id,
            'max_speed': t.speed,
            'origin_from_lane': t.origin_from_lane,
            'origin_to_lane': t.origin_to_lane,
            'dest_from_lane': t.destination_from_lane,
            'dest_to_lane': t.destination_to_lane
        }

    # load centroid connections
    for c in cen_connections:
        from_id = c.owner.id
        from_name = c.owner.name
        to_id = c.connection_object.id
        to_name = c.connection_object.name

        # invert from and to if connection is reversed
        if c.connection_type == 1:  # TODO verify this
            from_id, to_id = to_id, from_id
            from_name, to_name = to_name, from_name

        scenario_data['connections'][c.id] = {
            'from_id': from_id,
            'from_name': from_name,
            'to_id': to_id,
            'to_name': to_name
        }

    return scenario_data


# collect template path
file_path = os.path.join(config.PROJECT_PATH,
                         'flow/utils/aimsun/aimsun_template_path')
with open(file_path, 'r') as f:
    template_path = f.readline()
os.remove(file_path)

# open template in Aimsun
print('[load.py] Loading template ' + template_path)
model = AimsunTemplate(GKSystem, GKGUISystem)
model.load(template_path)

# collect the simulation parameters
params_file = 'flow/core/kernel/scenario/data.json'
params_path = os.path.join(config.PROJECT_PATH, params_file)
with open(params_path) as f:
    data = json.load(f)

# retrieve replication by name
replication_name = data['replication_name']
replication = model.find_by_name(model.replications, replication_name)

if replication is None:
    print('[load.py] ERROR: Replication ' + replication_name +
          ' does not exist.')

# retrieve experiment and scenario
experiment = replication.experiment
scenario = experiment.scenario
scenario_data = scenario.input_data
scenario_data.add_extension(os.path.join(
    config.PROJECT_PATH, 'flow/utils/aimsun/run.py'), True)

# if subnetwork_name was specified in the Aimsun params,
# try to only load subnetwork; it not specified or if
# subnetwork is not found, load the whole network
subnetwork_name = data['subnetwork_name']
if subnetwork_name is not None:
    subnetwork = model.find_by_name(model.problem_nets, subnetwork_name)
    if subnetwork:
        scenario_data = load_subnetwork(subnetwork, scenario)
    else:
        print('[load.py] ERROR: Subnetwork ' + subnetwork_name +
              ' could not be found. Loading the whole network.')
        scenario_data = load_network()
else:
    scenario_data = load_network()

# save template's scenario into a file to be loaded into Flow's scenario
scenario_data_file = 'flow/core/kernel/scenario/scenario_data.json'
scenario_data_path = os.path.join(config.PROJECT_PATH, scenario_data_file)
with open(scenario_data_path, 'w') as f:
    json.dump(scenario_data, f, sort_keys=True, indent=4)
    print('[load.py] Template\'s scenario data written into ' +
          scenario_data_path)

# create a check file to announce that we are done
# writing all the scenario data into the .json file
check_file = 'flow/core/kernel/scenario/scenario_data_check'
check_file_path = os.path.join(config.PROJECT_PATH, check_file)
open(check_file_path, 'a').close()

# get simulation step attribute column
col_sim = model.get_column('GKExperiment::simStepAtt')
# set new simulation step value
experiment.set_data_value(col_sim, data['sim_step'])
# run the simulation
model.run_replication(replication=replication, render=data['render'])
