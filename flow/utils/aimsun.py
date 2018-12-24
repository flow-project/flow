import sys

try:
    # Load user config if exists, else load default config
    import flow.config as config
except ImportError:
    import flow.config_default as config

# sys.path.append(config.AIMSUN_SITEPACKAGE)
SITEPACKAGES = "/home/aboudy/anaconda2/envs/aimsun/lib/python2.7/site-packages"
sys.path.append(SITEPACKAGES)

sys.path.append('/home/aboudy/Aimsun_Next_8_3_0/programming/Aimsun Next API/AAPIPython/Micro')

from copy import deepcopy
import argparse
import json
import os
import numpy
from numpy import pi, sin, cos, linspace


# Load an empty template
gui = GKGUISystem.getGUISystem().getActiveGui()
gui.newDoc("/home/aboudy/Downloads/Aimsun_Flow.ang", "EPSG:32601")
model = gui.getActiveModel()


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='FIXME',
        epilog='FIXME')

    # required input parameters
    parser.add_argument(
        'filename', type=str, help='Directory containing nodes and edges')
    return parser


def generate_net(nodes, edges, connections):
    # Aimsun GUI
    gui = GKGUISystem.getGUISystem().getActiveGui()

    lane_width = 3.6  # TODO additional params??

    # Draw edges
    for edge in edges:
        points = GKPoints()
        if "shape" in edge:
            for p in edge["shape"]:
                new_point = GKPoint()
                new_point.set(p[0], p[1], 0)
                points.append(new_point)

            cmd = model.createNewCmd(model.getType("GKSection"))
            cmd.setPoints(edge["numLanes"], lane_width, points)
            model.getCommander().addCommand(cmd)
            section = cmd.createdObject()
            section.setName(edge["id"])
            sectionType = model.getType("GKSection")
            edge_aimsun = model.getCatalog().findByName(edge["id"], sectionType)
            edge_aimsun.setSpeed(edge["speed"])
        else:
            new_point = GKPoint()
            first_node = next(node for node in nodes if node["id"] == edge["from"])
            new_point.set(first_node['x'], first_node['y'], 0)
            points.append(new_point)
            new_point = GKPoint()
            end_node = next(node for node in nodes if node["id"] == edge["to"])
            new_point.set(end_node['x'], end_node['y'], 0)
            points.append(new_point)

    # Draw turnings
    sectionType = model.getType("GKSection")

    for node in nodes:
        node_id = node['id']
        from_edges = [edge['id'] for edge in edges if edge['from'] == node_id]
        to_edges = [edge['id'] for edge in edges if edge['to'] == node_id]
        for i in range(len(from_edges)):
            for j in range(len(to_edges)):
                cmd = model.createNewCmd(model.getType("GKTurning"))
                to_section = model.getCatalog().findByName(
                    from_edges[i], sectionType, True)
                from_section = model.getCatalog().findByName(
                    to_edges[j], sectionType, True)
                cmd.setTurning(from_section, to_section)
                model.getCommander().addCommand(cmd)
                # turn = cmd.createdObject()
                # turn.setName("%s_to_%s" % (node["from"], node["to"]))

    # save doc
    gui.saveAs('flow.ang')

    replication_name = "Replication 870"
    replication = model.getCatalog().findByName(replication_name)
    GKSystem.getSystem().executeAction("execute", replication, [], "")


with open('/home/aboudy/Documents/flow/flow/core/kernel/scenario/data.json') as f:
    data = json.load(f)
nodes = data['nodes']
edges = data['edges']
types = data['types']

for i in range(len(edges)):
    if 'type' in edges[i]:
        for typ in types:
            if typ['id'] == edges[i]['type']:
                new_dict = deepcopy(typ)
                new_dict.pop("id")
                edges[i].update(new_dict)
            break
connections = data['connections']

generate_net(nodes, edges, connections)
