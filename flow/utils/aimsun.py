import sys
SITEPACKAGES = "/home/yashar/anaconda3/envs/aimsun/lib/python2.7/site-packages"
sys.path.append(SITEPACKAGES)

from copy import deepcopy
import argparse
import json
import os
from flow.core.params import NetParams, InFlows



import numpy
from numpy import pi, sin, cos, linspace


gui = GKGUISystem.getGUISystem().getActiveGui()
#     # Load an empty template
gui.newDoc("/home/yashar/Aimsun_Next_8_3_0/templates/Aimsun_Flow.ang",
           "EPSG:32601")
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


def generate_net(nodes, edges, net_params):
    # Aimsun GUI
    gui = GKGUISystem.getGUISystem().getActiveGui()

    lane_width = 3.6  # TODO additional params??
    num_lanes = net_params.additional_params["lanes"]
    # TODO: add speed limit

    # Draw edges
    for edge in edges:
        points = GKPoints()
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
                turn = cmd.createdObject()
                #turn.setName("%s_to_%s" % (node["from"], node["to"]))

    # save doc  # TODO: what is this? Can we make it the commented thing?
    gui.saveAs('/home/yashar/Documents/test_flow.ang')
    # gui.saveAs(os.path.join(os.path.dirname(__file__), 'flow.ang'))

    return 0


with open('/home/yashar/git_clone/flow/flow/core/kernel/scenario/data.json') as f:
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
# connections = data['connections']  # FIXME: add later

net_params = NetParams()
net_params.__dict__ = data["net_params"].copy()
net_params.inflows = InFlows()
if data["net_params"]["inflows"]:
    net_params.inflows.__dict__ = data["net_params"]["inflows"].copy()

generate_net(nodes, edges, net_params)
