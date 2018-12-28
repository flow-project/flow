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

    # double_edges = []
    # for i in range(len(edges)):
    #     for j in range(i, len(edges)):
    #         if edges[i]["from"] == edges[j]["to"] and edges[i]["to"] == edges[j]["from"]:
    #             double_edges.append(i)
    #             break
    # edges = [edges[i] for i in range(len(edges)) if i not in double_edges]

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
            for edg in edges:
                if edg["from"] == edge["to"] and edg["to"] == edge["from"]:
                    print (edge)
                    del_x = np.array([last_node['x'] - first_node['x']])
                    del_y = np.array([last_node['y'] - first_node['y']])
                    theta = np.arctan2(del_y, del_x) * 180 / np.pi
                    first_node_offset[0] += lane_width * 0.5 *\
                                          np.sin(theta * np.pi / 180)
                    first_node_offset[1] -= lane_width * 0.5 * \
                                          np.cos(theta * np.pi / 180)
                    last_node_offset[0] += lane_width * 0.5 *\
                                          np.sin(theta * np.pi / 180)
                    last_node_offset[1] -= lane_width * 0.5 *\
                                          np.cos(theta * np.pi / 180)

            new_point = GKPoint()
            new_point.set(first_node['x'] + first_node_offset[0],
                          first_node['y'] + first_node_offset[1],
                          0)
            points.append(new_point)
            new_point = GKPoint()
            end_node = next(node for node in nodes if node["id"] == edge["to"])
            new_point.set(last_node['x'] + last_node_offset[0],
                          last_node['y'] + last_node_offset[1],
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

        #if the node is a junction with a list of connections
        if len(to_edges) > 1 and len(from_edges) > 1 \
                and connections is not None: # TODO change this to connecctions[node['id']]
            # add connections
            for connection in connections:
                cmd = model.createNewCmd(type_turn)
                from_section = model.getCatalog().findByName(
                    connection["from"], type_section, True)
                to_section = model.getCatalog().findByName(
                    connection["to"], type_section, True)
                cmd.setTurning(from_section, to_section)
                model.getCommander().addCommand(cmd)
                turn = cmd.createdObject()
                turn_name = "{}_to_{}".format(connection["from"],
                                              connection["to"])
                turn.setName(turn_name)
                existing_node = turn.getNode()
                if existing_node is not None:
                    existing_node.removeTurning(turn)
                # add the turning to the node
                new_node.addTurning(turn, False, True)

        # if the node is not a junction or connections is None
        else:
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
                    existing_node = turn.getNode()
                    if existing_node is not None:
                        existing_node.removeTurning(turn)

                    # add the turning to the node
                    new_node.addTurning(turn, False, True)

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

    # Create new states based on vehicle types
    for veh_type in veh_types:
        new_state = create_state(model, veh_type["veh_id"])
        # find vehicle type
        veh_type = model.getCatalog().findByName(
            veh_type["veh_id"], model.getType("GKVehicle"))
        # set state vehicles
        new_state.setVehicle(veh_type)
        #set_state_vehicle(model, new_state, veh_type["veh_id"])
        # set_state_vehicle(model, veh_type["veh_id"], veh_type["veh_id"])

    # add traffic inflows to traffic states
    for inflow in inflows:
        print (inflow)
        traffic_state_aimsun = model.getCatalog().findByName(
            inflow["vtype"], type_traffic_state)
        edge_aimsun = model.getCatalog().findByName(
            inflow['edge'], type_section)
        traffic_state_aimsun.setEntranceFlow(
            edge_aimsun, None, inflow['vehsPerHour'])

    # set traffic demand
    for veh_type in veh_types:
        # find the state for each vehicle type
        state_car = model.getCatalog().findByName(
            veh_type["veh_id"], type_traffic_state)
        demand = model.getCatalog().findByName(
            "Traffic Demand 864", type_demand)
        if demand is not None and demand.isA("GKTrafficDemand"):
            # clear the demand of any previous item
            demand.removeSchedule()
            # Add the state
            if state_car != None and state_car.isA("GKTrafficState"):
                set_demand_item(model, demand, state_car)
            model.getCommander().addCommand(None)
        else:
            create_traffic_demand(model, veh_type["veh_id"])  # TODO debug

    # set the view to "whole world" in Aimsun
    view = gui.getActiveViewWindow().getView()
    if view is not None:
        view.wholeWorld()

    # set view mode, each vehicle type with different color
    set_vehicles_color(model)

    # save
    gui.saveAs('flow.ang')


def get_state_folder(model):
    folder_name = "GKModel::trafficStates"
    folder = model.getCreateRootFolder().findFolder(folder_name)
    if folder == None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folder_name)
    return folder


def create_state(model, name):
    state = GKSystem.getSystem().newObject("GKTrafficState", model)
    state.setName(name)
    folder = get_state_folder(model)
    folder.append(state)
    return state


def get_demand_folder(model):
    folder_name = "GKModel::trafficDemands"
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


def set_demand_item(model, demand, item):
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


def set_state_vehicle(model, state, veh_type_name):
    # find vehicle type
    veh_type = model.getCatalog().findByName(
        veh_type_name, model.getType("GKVehicle"))
    # set state vehicles
    state.setVehicle(veh_type)
    # find the state object
    # state_car = model.getCatalog().findByName(
    #     state_name, model.getType("GKTrafficState"))
    # state_car.setVehicle(veh_type)


def set_vehicles_color(model):
    viewMode = model.getGeoModel().findMode(
        "GKViewMode::VehiclesByVehicleType", False)
    if viewMode is None:
        viewMode = GKSystem.getSystem().newObject("GKViewMode", model)
        viewMode.setInternalName("GKViewMode::VehiclesByVehicleType")
        viewMode.setName("DYNAMIC: Simulation Vehicles by Vehicle Type")
        model.getGeoModel().addMode(viewMode)
    viewMode.removeAllStyles()
    viewStyle = model.getGeoModel().findStyle(
        "GKViewModeStyle::VehiclesByVehicleType")
    if viewStyle is None:
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
        vehicles = model.getCatalog().getObjectsByType(
            model.getType("GKVehicle"))
        if vehicles is not None:
            ramp.lines(len(vehicles))
            i = 0
            for vehicle in vehicles.itervalues():
                color_range = viewStyle.addRange(vehicle.getName())
                color_range.color = ramp.getColor(i)
                i = i + 1
        model.getGeoModel().addStyle(viewStyle)
    viewMode.addStyle(viewStyle)


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
