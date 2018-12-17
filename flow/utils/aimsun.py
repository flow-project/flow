import sys
SITEPACKAGES = "/home/yashar/anaconda3/envs/aimsun/lib/python2.7/site-packages"
sys.path.append(SITEPACKAGES)

import numpy
from numpy import pi, sin, cos, linspace


import argparse
import json
from flow.core.params import NetParams, InFlows



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









#
#
# def specify_types(net_params):
#   """See parent class."""
#   lanes = net_params.additional_params["lanes"]
#   speed_limit = net_params.additional_params["speed_limit"]
#
#   types = [{
#     "id": "edgeType",
#     "numLanes": repr(lanes),
#     "speed": repr(speed_limit)
#   }]
#
#   return types
#
#
# def specify_routes(net_params):
#   """See parent class."""
#   rts = {
#     "top": ["top", "left", "bottom", "right"],
#     "left": ["left", "bottom", "right", "top"],
#     "bottom": ["bottom", "right", "top", "left"],
#     "right": ["right", "top", "left", "bottom"]
#   }
#
#   return rts
#
# def specify_turnings(net_params):
#   """See parent class."""
#   turnings = {
#     "top_to_left": ["top", "left"],
#     "left_to_bottom": ["left", "bottom"],
#     "bottom_to_right": ["bottom", "right"],
#     "right_to_top": ["right", "top"]
#   }
#   return turnings
#
#
#
#
#
#
#
# class NetParams:
#     """Network configuration parameters.
#
#     Unlike most other parameters, NetParams may vary drastically dependent
#     on the specific network configuration. For example, for the ring road
#     the network parameters will include a characteristic length, number of
#     lanes, and speed limit.
#
#     In order to determine which additional_params variable may be needed
#     for a specific scenario, refer to the ADDITIONAL_NET_PARAMS variable
#     located in the scenario file.
#     """
#
#     def __init__(self,
#                  no_internal_links=True,
#                  inflows=None,
#                  in_flows=None,
#                  osm_path=None,
#                  netfile=None,
#                  additional_params=None):
#       """Instantiate NetParams.
#
#       Parameters
#       ----------
#       no_internal_links : bool, optional
#           determines whether the space between edges is finite. Important
#           when using networks with intersections; default is False
#       inflows : InFlows type, optional
#           specifies the inflows of specific edges and the types of vehicles
#           entering the network from these edges
#       osm_path : str, optional
#           path to the .osm file that should be used to generate the network
#           configuration files. This parameter is only needed / used if the
#           OpenStreetMapGenerator generator class is used.
#       netfile : str, optional
#           path to the .net.xml file that should be passed to SUMO. This is
#           only needed / used if the NetFileGenerator class is used, such as
#           in the case of Bay Bridge experiments (which use a custom net.xml
#           file)
#       additional_params : dict, optional
#           network specific parameters; see each subclass for a description of
#           what is needed
#       """
#       self.no_internal_links = no_internal_links
#       # if inflows is None:
#       #   self.inflows = InFlows()
#       # else:
#       #   self.inflows = inflows
#       self.osm_path = osm_path
#       self.netfile = netfile
#       self.additional_params = additional_params or {}
#       if in_flows is not None:
#         warnings.simplefilter("always", PendingDeprecationWarning)
#         warnings.warn(
#           "in_flows will be deprecated in a future release, use "
#           "inflows instead.",
#           PendingDeprecationWarning
#         )
#         self.inflows = in_flows


def generate_net(nodes, edges, net_params):
    # Aimsun GUI
    gui = GKGUISystem.getGUISystem().getActiveGui()

    lane_width = 3.6  # TODO additional params??
    num_lanes = net_params.additional_params["lanes"]

    # Draw edges
    for edge in edges:
        points = GKPoints()
        for p in edge["shape"]:
            new_point = GKPoint()
            new_point.set(p[0], p[1], p[2])
            points.append(new_point)

        cmd = model.createNewCmd(model.getType("GKSection"))
        cmd.setPoints(num_lanes, lane_width, points)
        model.getCommander().addCommand(cmd)
        section = cmd.createdObject()
        section.setName(edge["id"])

    # Draw turnings
    sectionType = model.getType("GKSection")

    for node in nodes:
        cmd = model.createNewCmd(model.getType("GKTurning"))
        from_section = model.getCatalog().findByName(node["from"],
                                                     sectionType,
                                                     True)
        to_section = model.getCatalog().findByName(node["to"],
                                                   sectionType,
                                                   True)
        cmd.setTurning(from_section, to_section)
        model.getCommander().addCommand(cmd)
        turn = cmd.createdObject()
        turn.setName("%s_to_%s" % (node["from"], node["to"]))

    # save doc
    gui.saveAs('/home/yashar/Documents/test_flow.ang')

    return 0


with open('/home/yashar/git_clone/flow/flow/core/kernel/scenario/data.json') as f:
    data = json.load(f)
nodes = data['nodes']
edges = data['edges']
print(data)

net_params = NetParams()
net_params.__dict__ = data["net_params"].copy()
net_params.inflows = InFlows()
if data["net_params"]["inflows"]:
    net_params.inflows.__dict__ = data["net_params"]["inflows"].copy()

generate_net(nodes, edges, net_params)
