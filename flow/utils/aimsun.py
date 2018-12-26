import sys

try:
    # Load user config if exists, else load default config
    import flow.config as config
except ImportError:
    import flow.config_default as config

#sys.path.append(config.AIMSUN_SITEPACKAGE)
SITEPACKAGES = "/home/yashar/anaconda3/envs/aimsun/lib/python2.7/site-packages"
sys.path.append(SITEPACKAGES)

sys.path.append('/home/yashar/Aimsun_Next_8_3_0/programming/Aimsun Next API/AAPIPython/Micro')

from copy import deepcopy
import argparse
import json
import os


import numpy
from numpy import pi, sin, cos, linspace


# Load an empty template
gui = GKGUISystem.getGUISystem().getActiveGui()
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


def generate_net(nodes, edges, connections):
    # Aimsun GUI
    gui = GKGUISystem.getGUISystem().getActiveGui()

    lane_width = 3.6  # TODO additional params??
    type_section = model.getType("GKSection")
    type_node = model.getType("GKNode")
    type_turn = model.getType("GKTurning")

    # # determine junctions
    # junctions = {}
    # for node in nodes:
    #     from_edges = [
    #         edge['id'] for edge in edges if edge['from'] == node['id']]
    #     to_edges = [edge['id'] for edge in edges if edge['to'] == node['id']]
    #     if len(to_edges) > 1 and len(from_edges) > 1:
    #         junctions[node['id']]["from_edges"] = from_edges
    #         junctions[node['id']]["to_edges"] = to_edges

    # draw edges
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
            edge_aimsun = model.getCatalog().findByName(
                edge["id"], type_section)
            edge_aimsun.setSpeed(edge["speed"])
        else:
            new_point = GKPoint()
            first_node = next(node for node in nodes
                              if node["id"] == edge["from"])
            first_node_offset = [0, 0]
            last_node_offset = [0, 0]
            if "radius" in edge:
                first_node_offset = edge["radius"][:2]
                last_node_offset = edge["radius"][2:]
            new_point.set(first_node['x'] + first_node_offset[0],
                          first_node['y'] + first_node_offset[1],
                          0)
            points.append(new_point)
            new_point = GKPoint()
            end_node = next(node for node in nodes if node["id"] == edge["to"])
            new_point.set(end_node['x'] + last_node_offset[0],
                          end_node['y'] + last_node_offset[1],
                          0)
            points.append(new_point)

            cmd = model.createNewCmd(type_section)
            cmd.setPoints(edge["numLanes"], lane_width, points)
            model.getCommander().addCommand(cmd)
            section = cmd.createdObject()
            section.setName(edge["id"])
            edge_aimsun = model.getCatalog().findByName(
                edge["id"], type_section)
            edge_aimsun.setSpeed(edge["speed"])

    # draw nodes and connections

    for node in nodes:
        # add a new node in Aimsun
        node_pos = GKPoint()
        node_pos.set(node['x'], node['y'], 0)
        cmd = model.createNewCmd(type_node)
        cmd.setPosition(node_pos)
        model.getCommander().addCommand(cmd)
        new_node = cmd.createdObject()
        new_node.setName(node["id"])
        node_aimsun = model.getCatalog().findByName(
            node["id"], type_node)
        # # list of edges from and to the node
        from_edges = [
            edge['id'] for edge in edges if edge['from'] == node['id']]
        to_edges = [edge['id'] for edge in edges if edge['to'] == node['id']]
        # if the node is a junction with a list of connections # TODO there is a bug here
        # if len(to_edges) > 1 and len(from_edges) > 1 \
        #         and connections is not None: # TODO change this to connecctions[node['id']]
        #     # add connections
        #     for connection in connections:
        #         cmd = model.createNewCmd(type_turn)
        #         from_section = model.getCatalog().findByName(
        #             connection["from"], type_section, True)
        #         to_section = model.getCatalog().findByName(
        #             connection["to"], type_section, True)
        #         cmd.setTurning(from_section, to_section)
        #         model.getCommander().addCommand(cmd)
        #         turn = cmd.createdObject()
        #         turn_name = "{}_to_{}".format(connection["from"],
        #                                       connection["to"])
        #         turn.setName(turn_name)
        #         turn_aimsun = model.getCatalog().findByName(
        #             turn_name, type_turn)
        #         # turn_aimsun.setNode(node_aimsun)
        #         node_aimsun.addTurning(turn_aimsun, False, True)
        # # if the node is not a junction or connections is None
        # else:
        #     for i in range(len(from_edges)):
        #         for j in range(len(to_edges)):
        #             cmd = model.createNewCmd(type_turn)
        #             to_section = model.getCatalog().findByName(
        #                 from_edges[i], type_section, True)
        #             from_section = model.getCatalog().findByName(
        #                 to_edges[j], type_section, True)
        #             cmd.setTurning(from_section, to_section)
        #             model.getCommander().addCommand(cmd)
        #             turn = cmd.createdObject()
        #             turn_name = "{}_to_{}".format(from_edges[i], to_edges[j])
        #             turn.setName(turn_name)
        #             turn_aimsun = model.getCatalog().findByName(
        #                 turn_name, type_turn)
        #             node_aimsun.addTurning(turn_aimsun, False, True)

        # TODO remove this if the bug is resolved
        # add all possible connections
        for i in range(len(from_edges)):
            for j in range(len(to_edges)):
                cmd = model.createNewCmd(type_turn)
                to_section = model.getCatalog().findByName(
                    from_edges[i], type_section, True)
                from_section = model.getCatalog().findByName(
                    to_edges[j], type_section, True)
                cmd.setTurning(from_section, to_section)
                model.getCommander().addCommand(cmd)
                turn = cmd.createdObject()
                turn_name = "{}_to_{}".format(from_edges[i], to_edges[j])
                turn.setName(turn_name)
                turn_aimsun = model.getCatalog().findByName(
                    turn_name, type_turn)
                #node_aimsun.addTurning(turn_aimsun, False, True)
        # remove the connections that shouldn't be in the model
        turn_aimsun = model.getCatalog().findByName(
            "right_lower_ring_out_to_bottom_upper_ring_in", type_turn)
        if turn_aimsun != None:
            cmd = turn_aimsun.getDelCmd()
            model.getCommander().addCommand(cmd)
        turn_aimsun = model.getCatalog().findByName(
            "bottom_upper_ring_out_to_right_lower_ring_in", type_turn)
        if turn_aimsun != None:
            cmd = turn_aimsun.getDelCmd()
            model.getCommander().addCommand(cmd)


    gui.saveAs('flow.ang')
    import AAPI as aimsun_api
    return aimsun_api


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
connections = data['connections']

kernel_api = generate_net(nodes, edges, connections)
