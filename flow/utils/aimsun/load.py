# flake8: noqa
import sys
import os
sys.path.append("/Users/nathan/projects/flow/")
import flow.config as config

SITEPACKAGES = os.path.join(config.AIMSUN_SITEPACKAGES,
                            "lib/python2.7/site-packages")
sys.path.append(SITEPACKAGES)

sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))

from flow.core.params import InFlows
from flow.core.params import TrafficLightParams

from copy import deepcopy
import json
import numpy as np


# Loads the whole network into a dictionary and returns it
def load_network():
    # get all relevant objects in network
    section_type = model.getType("GKSection")
    node_type = model.getType("GKNode")
    turning_type = model.getType("GKTurning")
    cen_connection_type = model.getType("GKCenConnection")

    sections = model.getCatalog().getObjectsByType(section_type).values()
    nodes = model.getCatalog().getObjectsByType(node_type).values()
    turnings = model.getCatalog().getObjectsByType(turning_type).values()
    cen_connections = model.getCatalog().getObjectsByType(cen_connection_type).values()

    scenario_data = get_dict_from_objects(sections, nodes, turnings, cen_connections)
    return scenario_data


# Loads subnetwork into a dictionary and returns it
def load_subnetwork(subnetwork, scenario):
    # get all objects in subnetwork
    objs = list(subnetwork.classifyObjects(scenario.getId()))

    sections = [o for o in objs if o.getTypeName() == 'GKSection']
    nodes = [o for o in objs if o.getTypeName() == 'GKNode']
    turnings = [o for o in objs if o.getTypeName() == 'GKTurning']
    cen_connections = [o for o in objs if o.getTypeName() == 'GKCenConnection']

    scenario_data = get_dict_from_objects(sections, nodes, turnings, cen_connections)
    return scenario_data


# Loads all the data into dictionaries
def get_dict_from_objects(sections, nodes, turnings, cen_connections):
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
    centroid_config = model.getCatalog().findByName(
        centroid_config_name, model.getType("GKCentroidConfiguration"))
    if not centroid_config:
        print("[load.py] ERROR: Centroid configuration "
              + centroid_config_name + " does not exist.")
    for c in centroid_config.getOriginCentroids():
        scenario_data['centroids'][c.getId()] = {'type': 'in'}
    for c in centroid_config.getDestinationCentroids():
        scenario_data['centroids'][c.getId()] = {'type': 'out'}

    # load sections
    for s in sections:
        scenario_data['sections'][s.getId()] = {
            'name': s.getName(),
            'numLanes': s.getNbFullLanes(),
            # FIXME this is a mean of the lanes lengths 
            # (bc they don't have to be all of the same size)
            # it may not be 100% accurate
            'length': s.getLanesLength2D() / s.getNbFullLanes(),
            'max_speed': s.getSpeed()
        }

    # load nodes
    for n in nodes:
        scenario_data['nodes'][n.getId()] = {
            'name': n.getName(),
            'nb_turnings': len(n.getTurnings())
        }

    # load turnings
    for t in turnings:
        scenario_data['turnings'][t.getId()] = {
            'name': t.getName(),
            'length': t.getPolygon().length2D() / 2, # FIXME not totally accurate
            'origin_section_name': t.getOrigin().getName(),
            'origin_section_id': t.getOrigin().getId(),
            'dest_section_name': t.getDestination().getName(),
            'dest_section_id': t.getDestination().getId(),
            'node_id': t.getNode().getId(),
            'max_speed': t.getSpeed(),
            'origin_from_lane': t.getOriginFromLane(),
            'origin_to_lane': t.getOriginToLane(),
            'dest_from_lane': t.getDestinationFromLane(),
            'dest_to_lane': t.getDestinationToLane()
        }

    # load centroid connections
    for c in cen_connections:
        from_id = c.getOwner().getId()
        from_name = c.getOwner().getName()
        to_id = c.getConnectionObject().getId()
        to_name = c.getConnectionObject().getName()

        # invert from and to if connection is reversed
        if c.getConnectionType() == 1:  # TODO verify this
            from_id, to_id = to_id, from_id
            from_name, to_name = to_name, from_name

        scenario_data['connections'][c.getId()] = {
            'from_id': from_id,
            'from_name': from_name,
            'to_id': to_id,
            'to_name': to_name
        }

    return scenario_data


# collect template path
file_path = os.path.join(config.PROJECT_PATH,
                         "flow/utils/aimsun/aimsun_template_path")
with open(file_path, 'r') as f:
    template_path = f.readline()
os.remove(template_path)

# open template in Aimsun
print("[load.py] Loading template " + template_path)
gui = GKGUISystem.getGUISystem().getActiveGui()
gui.loadNetwork(template_path)
model = gui.getActiveModel()

# collect the simulation parameters
params_file = 'flow/core/kernel/scenario/data.json'
params_path = os.path.join(config.PROJECT_PATH, params_file)
with open(params_path) as f:
    data = json.load(f)

# retrieve replication by name
replication_name = data["replication_name"]
replication = model.getCatalog().findByName(
    replication_name, model.getType("GKReplication"))

if not replication:
    print("[load.py] ERROR: Replication " + replication_name + " does not exist.")

# retrieve experiment and scenario
experiment = replication.getExperiment()
scenario = experiment.getScenario()
scenario_data = scenario.getInputData()
scenario_data.addExtension(os.path.join(
    config.PROJECT_PATH, "flow/utils/aimsun/run.py"), True)

# if subnetwork_name was specified in the Aimsun params,
# try to only load subnetwork; it not specified or if
# subnetwork is not found, load the whole network
subnetwork_name = data['subnetwork_name']
if subnetwork_name is not None:
    subnetwork = model.getCatalog().findByName(
        subnetwork_name, model.getType("GKProblemNet"))
    if subnetwork:
        scenario_data = load_subnetwork(subnetwork, scenario)
    else:
        print("[load.py] ERROR: Subnetwork " + subnetwork_name
              + " could not be found. Loading the whole network.")
        scenario_data = load_network()
else:
    scenario_data = load_network()

# save template's scenario into a file to be loaded into Flow's scenario
scenario_data_file = "flow/core/kernel/scenario/scenario_data.json"
scenario_data_path = os.path.join(config.PROJECT_PATH, scenario_data_file)
with open(scenario_data_path, "w") as f:
    json.dump(scenario_data, f, sort_keys=True, indent=4) 
    print("[load.py] Template's scenario data written into "
          + scenario_data_path)

# create a check file to announce that we are done
# writing all the scenario data into the .json file
check_file = "flow/core/kernel/scenario/scenario_data_check"
check_file_path = os.path.join(config.PROJECT_PATH, check_file)
open(check_file_path, 'a').close()

# get simulation step attribute column
col_sim = model.getColumn('GKExperiment::simStepAtt')
# set new simulation step value
experiment.setDataValue(col_sim, data["sim_step"])
# run the simulation
# execute, "play": run with GUI, "execute": run in batch mode
mode = 'play' if data['render'] is True else 'execute'
GKSystem.getSystem().executeAction(mode, replication, [], "")
