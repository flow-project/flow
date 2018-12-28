import sys
import os
import flow.config as config

SITEPACKAGES = os.path.join(config.AIMSUN_SITEPACKAGES,
                            "lib/python2.7/site-packages")
sys.path.append(SITEPACKAGES)

sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))

from flow.core.params import InFlows
from copy import deepcopy
import argparse
import json
import numpy as np


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


def generate_net(nodes, edges, connections, inflows, veh_types):
    inflows = inflows.get()
    # Aimsun GUI
    gui = GKGUISystem.getGUISystem().getActiveGui()

    lane_width = 3.6  # TODO additional params??
    type_section = model.getType("GKSection")
    type_node = model.getType("GKNode")
    type_turn = model.getType("GKTurning")
    type_traffic_state = model.getType("GKTrafficState")
    type_vehicle = model.getType("GKVehicle")
    type_demand = model.getType("GKTrafficDemand")



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
            for p in edge["shape"]: #TODO add x, y offset (radius)
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
            first_node = next(node for node in nodes
                              if node["id"] == edge["from"])
            last_node = next(node for node in nodes
                             if node["id"] == edge["to"])
            first_node_offset = [0, 0]  # x, and y offset
            last_node_offset = [0, 0] # x, and y offset
            if "radius" in first_node:
                del_x = np.array([last_node['x'] - first_node['x']])
                del_y = np.array([last_node['y'] - first_node['y']])
                theta = np.arctan2(del_y, del_x) * 180 / np.pi
                first_node_offset[0] = first_node["radius"] *\
                                       np.cos(theta*np.pi/180)
                first_node_offset[1] = first_node["radius"] * \
                                       np.sin(theta*np.pi/180)

            if "radius" in last_node:
                del_x = np.array([last_node['x'] - first_node['x']])
                del_y = np.array([last_node['y'] - first_node['y']])
                theta = np.arctan2(del_y, del_x) * 180 / np.pi
                last_node_offset[0] = - last_node["radius"] * \
                                      np.cos(theta*np.pi/180)
                last_node_offset[1] = - last_node["radius"] * \
                                      np.sin(theta*np.pi/180)
            new_point = GKPoint()
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
        if turn_aimsun is not None:
            cmd = turn_aimsun.getDelCmd()
            model.getCommander().addCommand(cmd)
        turn_aimsun = model.getCatalog().findByName(
            "bottom_upper_ring_out_to_right_lower_ring_in", type_turn)
        if turn_aimsun is not None:
            cmd = turn_aimsun.getDelCmd()
            model.getCommander().addCommand(cmd)

    #
    def get_state_folder(model):
        folder_name = "GKModel::trafficStates"
        folder = model.getCreateRootFolder().findFolder(folder_name)
        if folder == None:
            folder = GKSystem.getSystem().createFolder(
                model.getCreateRootFolder(), folder_name)
        return folder

    # Creates a traffic state with the given name
    #
    def create_state(model, name):
        state = GKSystem.getSystem().newObject("GKTrafficState", model)
        state.setName(name)
        folder = get_state_folder(model)
        folder.append(state)
        return state

    def get_demand_folder(model):
        folder_name = "GKModel::trafficDemandsYahahahaha"
        folder = model.getCreateRootFolder().findFolder(folder_name)
        if folder == None:
            folder = GKSystem.getSystem().createFolder(
                model.getCreateRootFolder(), folder_name)
        return folder

    def create_traffic_demand(model, name):
        demand = GKSystem.getSystem().newObject("GKTrafficDemand", model)
        demand.setName(name)
        folder = get_demand_folder(model)
        folder.append(demand)
        return demand

    def setDemandItem(model, demand, item):
        if item.getVehicle() == None:
            model.getLog().addError("Invalid Demand Item: no vehicle")
        else:
            schedule = GKScheduleDemandItem()
            schedule.setTrafficDemandItem(item)
            # Starts at 8:00:00 AM
            schedule.setFrom(8 * 3600)
            # Duration: 500 hour
            schedule.setDuration(500 * 3600)
            demand.addToSchedule(schedule)


    # Create new states based on vehicle types
    for veh_type in veh_types:
        new_state = create_state(model, veh_type["veh_id"])

    # Create new vehicle demand on vehicle types
    demand_name = "Traffic Demand Test"
    demand_object = create_traffic_demand(model, demand_name)

    # Traffic State
    for veh_type in veh_types:
        # find the state for each vehicle type
        state_car = model.getCatalog().findByName(
            veh_type["veh_id"], type_traffic_state)
        demand = model.getCatalog().findByName(
            demand_name, type_demand)
        if demand is not None and demand.isA("GKTrafficDemand"):
            # clear the demand of any previous item
            demand.removeSchedule()
            # Add the state
            if state_car != None and state_car.isA("GKTrafficState"):
                setDemandItem(model, demand, state_car)
            model.getCommander().addCommand(None)
        else:
            print
            "Demand does not exist"

    model.getCommander().addCommand(None)


    # set vehicle types
    vehicles = model.getCatalog().getObjectsByType(type_vehicle)
    if vehicles is not None:
        for vehicle in vehicles.itervalues():
            name = vehicle.getName()
            if name == "Car":
                for veh_type in veh_types:
                    print (veh_type)
                    cmd = GKObjectDuplicateCmd()
                    cmd.init(vehicle)
                    model.getCommander().addCommand(cmd)
                    new_veh = cmd.createdObject()
                    new_veh.setName(veh_type["veh_id"])





    # add traffic inflow to traffic state
    traffic_state_aimsun = model.getCatalog().findByName(
        "state Car: 00:00 ", type_traffic_state)
    for inflow in inflows:
        edge_aimsun = model.getCatalog().findByName(
            inflow['edge'], type_section)
        traffic_state_aimsun.setEntranceFlow(
            edge_aimsun, None, inflow['vehsPerHour'])

    # set the view to "whole world" in Aimsun
    view = gui.getActiveViewWindow().getView()
    if view is not None:
        view.wholeWorld()

    # # set view mode, each vehicle type with different color
    viewMode = model.getGeoModel().findMode(
        "GKViewMode::VehiclesByVehicleType", False)
    if viewMode == None:
        viewMode = GKSystem.getSystem().newObject("GKViewMode", model)
        viewMode.setInternalName("GKViewMode::VehiclesByVehicleType")
        viewMode.setName("DYNAMIC: Simulation Vehicles by Vehicle Type")
        model.getGeoModel().addMode(viewMode)
    viewMode.removeAllStyles()
    viewStyle = model.getGeoModel().findStyle(
        "GKViewModeStyle::VehiclesByVehicleType")
    if viewStyle == None:
        viewStyle = GKSystem.getSystem().newObject("GKViewModeStyle", model)
        viewStyle.setInternalName("GKViewModeStyle::VehiclesByVehicleType")
        viewStyle.setName("DYNAMIC: Simulation Vehicles by Vehicle Type")
        viewStyle.setStyleType(GKViewModeStyle.eColor)
        viewStyle.setVariableType(GKViewModeStyle.eDiscrete)
        simType = model.getType("GKSimVehicle")
        typeColumn = simType.getColumn("GKSimVehicle::vehicleTypeAtt",
                                       GKType.eSearchOnlyThisType)
        viewStyle.setColumn(simType, typeColumn)
        ramp = GKColorRamp()
        ramp.setType(GKColorRamp.eRGB)
        vehicles = model.getCatalog().getObjectsByType(type_vehicle)
        if vehicles != None:
            ramp.lines(len(vehicles))
            i = 0
            for vehicle in vehicles.itervalues():
                color_range = viewStyle.addRange(vehicle.getName())
                color_range.color = ramp.getColor(i)
                i = i + 1
        model.getGeoModel().addStyle(viewStyle)

    viewMode.addStyle(viewStyle)
    model.getCommander().addCommand(None)

    # save
    gui.saveAs('flow.ang')


# collect the scenario-specific data
data_file = 'flow/core/kernel/scenario/data.json'
with open(os.path.join(config.PROJECT_PATH, data_file)) as f:
    data = json.load(f)

# export the data from the dictionary
nodes = data['nodes']
edges = data['edges']
types = data['types']
connections = data['connections']
veh_types = data['vehicle_types']

for i in range(len(edges)):
    if 'type' in edges[i]:
        for typ in types:
            if typ['id'] == edges[i]['type']:
                new_dict = deepcopy(typ)
                new_dict.pop("id")
                edges[i].update(new_dict)
                break

if data['inflows'] is not None:
    inflows = InFlows()
    inflows.__dict__ = data['inflows'].copy()
else:
    inflows = None

# generate the network
generate_net(nodes, edges, connections, inflows, veh_types)
