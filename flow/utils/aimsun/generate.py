# flake8: noqa
"""Script for generating a custom Aimsun network."""
# TODO adapt this file with respect to scripting_api.py

import sys
import os
import flow.config as config

SITEPACKAGES = os.path.join(config.AIMSUN_SITEPACKAGES,
                            "lib/python2.7/site-packages")
sys.path.append(SITEPACKAGES)

from flow.core.params import InFlows
from flow.core.params import TrafficLightParams

from copy import deepcopy
import json
import numpy as np


sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))


# Load an empty template
gui = GKGUISystem.getGUISystem().getActiveGui()
gui.newDoc(os.path.join(config.PROJECT_PATH,
                        "flow/utils/aimsun/Aimsun_Flow.ang"),
           "EPSG:32601")
model = gui.getActiveModel()


def generate_net(nodes,
                 edges,
                 connections,
                 inflows,
                 veh_types,
                 traffic_lights):
    """Generate a network in the Aimsun template.

    Parameters
    ----------
    nodes : list of dict
        all available nodes
    edges : list of dict
        all available edges
    connections : list of dict
        all available connections
    inflows : flow.core.params.InFlows
        the flow inflow object
    veh_types : list of dict
        list of vehicle types and their corresponding properties
    traffic_lights : flow.core.params.TrafficLightParams
        traffic light specific parameters
    """
    inflows = inflows.get()
    lane_width = 3.6  # TODO additional params??
    type_section = model.getType("GKSection")
    type_node = model.getType("GKNode")
    type_turn = model.getType("GKTurning")
    type_traffic_state = model.getType("GKTrafficState")
    type_vehicle = model.getType("GKVehicle")
    type_demand = model.getType("GKTrafficDemand")

    # draw edges
    for edge in edges:
        points = GKPoints()
        if "shape" in edge:
            for p in edge["shape"]:  # TODO add x, y offset (radius)
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
            edge_aimsun.setSpeed(edge["speed"] * 3.6)
        else:
            first_node, last_node = get_edge_nodes(edge, nodes)
            theta = get_edge_angle(first_node, last_node)
            first_node_offset = [0, 0]  # x, and y offset
            last_node_offset = [0, 0]  # x, and y offset

            # offset edge ends if there is a radius in the node
            if "radius" in first_node:
                first_node_offset[0] = first_node["radius"] * \
                    np.cos(theta*np.pi/180)
                first_node_offset[1] = first_node["radius"] * \
                    np.sin(theta*np.pi/180)
            if "radius" in last_node:
                last_node_offset[0] = - last_node["radius"] * \
                    np.cos(theta*np.pi/180)
                last_node_offset[1] = - last_node["radius"] * \
                    np.sin(theta*np.pi/180)

            # offset edge ends if there are multiple edges between nodes
            # find the edges that share the first node
            edges_shared_node = [edg for edg in edges
                                 if first_node["id"] == edg["to"] or
                                 last_node["id"] == edg["from"]]
            for new_edge in edges_shared_node:
                new_first_node, new_last_node = get_edge_nodes(new_edge, nodes)
                new_theta = get_edge_angle(new_first_node, new_last_node)
                if new_theta == theta - 180 or new_theta == theta + 180:
                    first_node_offset[0] += lane_width * 0.5 *\
                        np.sin(theta * np.pi / 180)
                    first_node_offset[1] -= lane_width * 0.5 * \
                        np.cos(theta * np.pi / 180)
                    last_node_offset[0] += lane_width * 0.5 *\
                        np.sin(theta * np.pi / 180)
                    last_node_offset[1] -= lane_width * 0.5 *\
                        np.cos(theta * np.pi / 180)
                    break

            new_point = GKPoint()
            new_point.set(first_node['x'] + first_node_offset[0],
                          first_node['y'] + first_node_offset[1],
                          0)
            points.append(new_point)
            new_point = GKPoint()
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
            edge_aimsun.setSpeed(edge["speed"] * 3.6)

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

        # list of edges from and to the node
        from_edges = [
            edge['id'] for edge in edges if edge['from'] == node['id']]
        to_edges = [edge['id'] for edge in edges if edge['to'] == node['id']]

        # if the node is a junction with a list of connections
        if len(to_edges) > 1 and len(from_edges) > 1 \
                and connections[node['id']] is not None:
            # add connections
            for connection in connections[node['id']]:
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

    # get the control plan
    control_plan = model.getCatalog().findByName(
            "Control Plan", model.getType("GKControlPlan"))

    # add traffic lights
    tls_properties = traffic_lights.get_properties()
    # determine junctions
    junctions = get_junctions(nodes)
    # add meters for all nodes in junctions
    for node in junctions:
        phases = tls_properties[node['id']]["phases"]
        print(phases)
        create_node_meters(model, control_plan, node['id'], phases)

    # set vehicle types
    vehicles = model.getCatalog().getObjectsByType(type_vehicle)
    if vehicles is not None:
        for vehicle in vehicles.itervalues():
            name = vehicle.getName()
            if name == "Car":
                for veh_type in veh_types:
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

    # add traffic inflows to traffic states
    for inflow in inflows:
        traffic_state_aimsun = model.getCatalog().findByName(
            inflow["vtype"], type_traffic_state)
        edge_aimsun = model.getCatalog().findByName(
            inflow['edge'], type_section)
        traffic_state_aimsun.setEntranceFlow(
            edge_aimsun, None, inflow['vehsPerHour'])

    # get traffic demand
    demand = model.getCatalog().findByName(
        "Traffic Demand 864", type_demand)
    # clear the demand of any previous item
    demand.removeSchedule()

    # set traffic demand
    for veh_type in veh_types:
        # find the state for each vehicle type
        state_car = model.getCatalog().findByName(
            veh_type["veh_id"], type_traffic_state)
        if demand is not None and demand.isA("GKTrafficDemand"):
            # Add the state
            if state_car is not None and state_car.isA("GKTrafficState"):
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

    # set API
    scenario_name = data["scenario_name"]
    scenario = model.getCatalog().findByName(
        scenario_name, model.getType("GKScenario"))  # find scenario
    scenario_data = scenario.getInputData()
    scenario_data.addExtension(os.path.join(
        config.PROJECT_PATH, "flow/utils/aimsun/run.py"), True)

    # save
    gui.saveAs('flow.ang')


def generate_net_osm(file_name, inflows, veh_types):
    """Generate a network from an osm file.

    Parameters
    ----------
    file_name : str
        path to the osm file
    inflows : flow.core.params.InFlows
        the flow inflow object
    veh_types : list of dict
        list of vehicle types and their corresponding properties
    """
    inflows = inflows.get()

    type_section = model.getType("GKSection")
    type_traffic_state = model.getType("GKTrafficState")
    type_vehicle = model.getType("GKVehicle")
    type_demand = model.getType("GKTrafficDemand")

    # load OSM file
    layer = None
    point = GKPoint()
    point.set(0, 0, 0)
    box = GKBBox()
    box.set(-1000, -1000, 0, 1000, 1000, 0)

    model.importFile(file_name, layer, point, box)

    # set vehicle types
    vehicles = model.getCatalog().getObjectsByType(type_vehicle)
    if vehicles is not None:
        for vehicle in vehicles.itervalues():
            name = vehicle.getName()
            if name == "Car":
                for veh_type in veh_types:
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

    # add traffic inflows to traffic states
    if inflows is not None:
        for inflow in inflows:
            traffic_state_aimsun = model.getCatalog().findByName(
                inflow["vtype"], type_traffic_state)
            edge_aimsun = model.getCatalog().findByName(
                inflow['edge'], type_section)
            traffic_state_aimsun.setEntranceFlow(
                edge_aimsun, None, inflow['vehsPerHour'])

    # get traffic demand
    demand = model.getCatalog().findByName(
        "Traffic Demand 864", type_demand)
    # clear the demand of any previous item
    demand.removeSchedule()

    # set traffic demand
    for veh_type in veh_types:
        # find the state for each vehicle type
        state_car = model.getCatalog().findByName(
            veh_type["veh_id"], type_traffic_state)
        if demand is not None and demand.isA("GKTrafficDemand"):
            # Add the state
            if state_car is not None and state_car.isA("GKTrafficState"):
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

    # set API
    scenario_name = data["scenario_name"]
    scenario = model.getCatalog().findByName(
        scenario_name, model.getType("GKScenario"))  # find scenario
    scenario_data = scenario.getInputData()
    scenario_data.addExtension(os.path.join(
        config.PROJECT_PATH, "flow/utils/aimsun/run.py"), True)

    # save
    gui.saveAs('flow.ang')


def get_junctions(nodes):
    """Return the nodes with traffic lights.

    Parameters
    ----------
    nodes : list of dict
        all available nodes

    Returns
    -------
    list of dict
        the nodes with traffic lights
    """
    junctions = []  # TODO check
    for node in nodes:
        if "type" in node:
            if node["type"] == "traffic_light":
                junctions.append(node)
    return junctions


def get_edge_nodes(edge, nodes):
    """Get first and last nodes of an edge.

    Parameters
    ----------
    edge : dict
        the edge information
    nodes : list of dict
        all available nodes

    Returns
    -------
    dict
        information on the first node
    dict
        information on the last node
    """
    first_node = next(node for node in nodes
                      if node["id"] == edge["from"])
    last_node = next(node for node in nodes
                     if node["id"] == edge["to"])
    return first_node, last_node


def get_edge_angle(first_node, last_node):
    """Compute the edge angle.

    Parameters
    ----------
    first_node : dict
        information on the first node
    last_node : dict
        information on the last node

    Returns
    -------
    float
        edge angle
    """
    del_x = np.array([last_node['x'] - first_node['x']])
    del_y = np.array([last_node['y'] - first_node['y']])
    theta = np.arctan2(del_y, del_x) * 180 / np.pi
    return theta


def get_state_folder(model):
    """Return traffic state folder.

    If the folder doesn't exist, a new folder will be created.

    Parameters
    ----------
    model : GKModel
        Aimsun model object

    Returns
    -------
    GKFolder
        an Aimsun folder object which contains traffic state.
    """
    folder_name = "GKModel::trafficStates"
    folder = model.getCreateRootFolder().findFolder(folder_name)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folder_name)
    return folder


def create_state(model, name):
    """Create a traffic state object.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    name : str
        name of the traffic state

    Returns
    -------
    GKTrafficState
        an Aimsun traffic state object
    """
    state = GKSystem.getSystem().newObject("GKTrafficState", model)
    state.setName(name)
    folder = get_state_folder(model)
    folder.append(state)
    return state


def get_demand_folder(model):
    """Return traffic demand folder.

    If the folder doesn't exist, a new folder will be created.

    Parameters
    ----------
    model : GKModel
        Aimsun model object

    Returns
    -------
    GKFolder
        an Aimsun folder object which contains traffic demand.
    """
    folder_name = "GKModel::trafficDemands"
    folder = model.getCreateRootFolder().findFolder(folder_name)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(
            model.getCreateRootFolder(), folder_name)
    return folder


def create_traffic_demand(model, name):
    """Create a traffic demand object.

    If the folder doesn't exist, a new folder will be created.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    name : str
        name of the traffic state

    Returns
    -------
    GKTrafficDemand
        an Aimsun traffic demand object
    """
    demand = GKSystem.getSystem().newObject("GKTrafficDemand", model)
    demand.setName(name)
    folder = get_demand_folder(model)
    folder.append(demand)
    return demand


def set_demand_item(model, demand, item):
    """Set a traffic demand item.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    demand : GKTrafficDemand
        an Aimsun traffic demand object
    item : GKTrafficDemandItem
        a traffic item which is valid for a vehicle type and a time interval
    """
    if item.getVehicle() is None:
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
    """Set state vehicle type.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    state : GKTrafficState
        an Aimsun traffic state object
    veh_type_name : str
        name of the vehicle type
    """
    # find vehicle type
    veh_type = model.getCatalog().findByName(
        veh_type_name, model.getType("GKVehicle"))
    # set state vehicles
    state.setVehicle(veh_type)


def set_vehicles_color(model):
    """Set view mode and view style.

    View mode and view style are used to show different vehicle types with
    different colors. View mode and view style are named
    "DYNAMIC: Simulation Vehicles by Vehicle Type".

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    """
    view_mode = model.getGeoModel().findMode(
        "GKViewMode::VehiclesByVehicleType", False)
    if view_mode is None:
        view_mode = GKSystem.getSystem().newObject("GKViewMode", model)
        view_mode.setInternalName("GKViewMode::VehiclesByVehicleType")
        view_mode.setName("DYNAMIC: Simulation Vehicles by Vehicle Type")
        model.getGeoModel().addMode(view_mode)
    view_mode.removeAllStyles()
    view_style = model.getGeoModel().findStyle(
        "GKViewModeStyle::VehiclesByVehicleType")
    if view_style is None:
        view_style = GKSystem.getSystem().newObject("GKViewModeStyle", model)
        view_style.setInternalName("GKViewModeStyle::VehiclesByVehicleType")
        view_style.setName("DYNAMIC: Simulation Vehicles by Vehicle Type")
        view_style.setStyleType(GKViewModeStyle.eColor)
        view_style.setVariableType(GKViewModeStyle.eDiscrete)
        sim_type = model.getType("GKSimVehicle")
        type_col = sim_type.getColumn("GKSimVehicle::vehicleTypeAtt",
                                      GKType.eSearchOnlyThisType)
        view_style.setColumn(sim_type, type_col)
        ramp = GKColorRamp()
        ramp.setType(GKColorRamp.eRGB)
        vehicles = model.getCatalog().getObjectsByType(
            model.getType("GKVehicle"))
        if vehicles is not None:
            ramp.lines(len(vehicles))
            for i, vehicle in enumerate(vehicles.itervalues()):
                color_range = view_style.addRange(vehicle.getName())
                color_range.color = ramp.getColor(i)
        model.getGeoModel().addStyle(view_style)
    view_mode.addStyle(view_style)


def get_control_plan_folder(model):
    """Return control plan folder.

    If the folder doesn't exist, a new folder will be created.

    Parameters
    ----------
    model : GKModel
        Aimsun model object

    Returns
    -------
    GKFolder
        an Aimsun folder object which contains control plan.
    """
    folder_name = "GKModel::controlPlans"
    folder = model.getCreateRootFolder().findFolder(folder_name)
    if folder is None:
        folder = GKSystem.getSystem().createFolder(model.getCreateRootFolder(),
                                                   folder_name)
    return folder


def create_control_plan(model, name):
    """Create a traffic control plan object.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    name : str
        name of the control plan

    Returns
    -------
    GKControlPlan
        an Aimsun control plan object
    """
    control_plan = GKSystem.getSystem().newObject("GKControlPlan", model)
    control_plan.setName(name)
    folder = get_control_plan_folder(model)
    folder.append(control_plan)
    return control_plan


def create_meter(model, edge):
    """Create a metering object.

    Parameters
    ----------
    model : GKModel
        Aimsun model object
    edge : str
        name of the edge

    Returns
    -------
    GKSectionObject
        an Aimsun metering (section object) object
    """
    section = model.getCatalog().findByName(edge, model.getType("GKSection"))
    meter_length = 2
    pos = section.getLanesLength2D() - meter_length
    type = model.getType("GKMetering")
    cmd = model.createNewCmd(model.getType("GKSectionObject"))
    # TODO double check the zeros
    cmd.init(type, section, 0, 0, pos, meter_length)
    model.getCommander().addCommand(cmd)
    meter = cmd.createdObject()
    meter.setName("meter_{}".format(section.getName()))
    return meter


def set_metering_times(
        cp, meter, cycle, green, yellow, offset, min_green, max_green):
    """Set a meter timing plan.

    Parameters
    ----------
    cp : GKControlPlan
        an aimsun control plan object
    meter : GKSectionObject
        an Aimsun metering (section object) object
    cycle : int
        cycle length
    green : int
        green phase duration
    yellow : int
        yellow phase duration
    offset : int
        offset duration
    min_green : int
        minimum green phase duration
    max_green : int
        maximum green phase duration
    """
    cp_meter = cp.createControlMetering(meter)
    cp_meter.setControlMeteringType(GKControlMetering.eExternal)
    cp_meter.setCycle(cycle)
    cp_meter.setGreen(green)
    cp_meter.setYellowTime(yellow)
    cp_meter.setOffset(offset)
    cp_meter.setMinGreen(min_green)
    cp_meter.setMaxGreen(max_green)


def create_node_meters(model, cp, node_id, phases):
    """Create meters for a node.

    Parameters
    ----------
    model:
    cp : GKControlPlan
        an aimsun control plan object
    node_id : str
        node ID
    phases :  list  of dict
        list of phases to be followed by the traffic light

    Returns
    -------
    list of GKSectionObject
        list of meters in the node
    """
    meters = []
    signal_groups = {}
    for connection in connections[node_id]:
        if connection["signal_group"] in signal_groups:
            signal_groups[
                connection["signal_group"]].append(connection["from"])
        else:
            signal_groups[connection["signal_group"]] = [connection["from"]]

    # get cycle length
    cycle = 0
    for signal_group, edges in signal_groups.items():
        cycle += int(phases[signal_group]["duration"]) + \
                 int(phases[signal_group]["yellow"])

    # set a meter for each edge in each signal group cycle length
    sum_phases = 0
    for signal_group, edges in signal_groups.items():
        green = int(phases[signal_group]["duration"])
        yellow = int(phases[signal_group]["yellow"])
        min_green = int(phases[signal_group]["minDur"])
        max_green = int(phases[signal_group]["maxDur"])
        for edge in edges:
            meter = create_meter(model, edge)
            set_metering_times(cp, meter, cycle, green, yellow,
                               sum_phases, min_green, max_green)
            meters.append(meter)
        sum_phases += green + yellow
    return meters


def set_sim_step(experiment, sim_step):
    """Set the simulation step of an Aimsun experiment.

    Parameters
    ----------
    experiment : GKTExperiment
        the experiment object
    sim_step : float
        desired simulation step
    """
    # Get Simulation Step attribute column
    col_sim = model.getColumn('GKExperiment::simStepAtt')
    # Set new simulation step value
    experiment.setDataValue(col_sim, sim_step)


# collect the scenario-specific data
data_file = 'flow/core/kernel/scenario/data.json'
with open(os.path.join(config.PROJECT_PATH, data_file)) as f:
    data = json.load(f)

# export the data from the dictionary
veh_types = data['vehicle_types']
osm_path = data['osm_path']

if data['inflows'] is not None:
    inflows = InFlows()
    inflows.__dict__ = data['inflows'].copy()
else:
    inflows = None

if data['traffic_lights'] is not None:
    traffic_lights = TrafficLightParams()
    traffic_lights.__dict__ = data['traffic_lights'].copy()
else:
    traffic_lights = None

# generate the network
if osm_path is not None:
    generate_net_osm(osm_path, inflows, veh_types)
    edge_osm = {}

    section_type = model.getType("GKSection")
    for types in model.getCatalog().getUsedSubTypesFromType(section_type):
        for s in types.itervalues():
            s_id = s.getId()
            num_lanes = s.getNbFullLanes()
            length = s.getLanesLength2D()
            speed = s.getSpeed()
            edge_osm[s_id] = {"speed": speed,
                              "length": length,
                              "numLanes": num_lanes}
    with open(os.path.join(config.PROJECT_PATH,
                           'flow/utils/aimsun/osm_edges.json'), 'w') \
            as outfile:
        json.dump(edge_osm, outfile, sort_keys=True, indent=4)

else:
    nodes = data['nodes']
    edges = data['edges']
    types = data['types']
    connections = data['connections']

    for i in range(len(edges)):
        if 'type' in edges[i]:
            for typ in types:
                if typ['id'] == edges[i]['type']:
                    new_dict = deepcopy(typ)
                    new_dict.pop("id")
                    edges[i].update(new_dict)
                    break
    generate_net(nodes, edges, connections, inflows, veh_types, traffic_lights)

# set sim step
sim_step = data["sim_step"]
# retrieve experiment by name
experiment_name = data["experiment_name"]
experiment = model.getCatalog().findByName(
    experiment_name, model.getType("GKTExperiment"))
set_sim_step(experiment, sim_step)

# run the simulation
# find the replication
replication_name = data["replication_name"]
replication = model.getCatalog().findByName(replication_name)
# execute, "play": run with GUI, "execute": run in batch mode
mode = 'play' if data['render'] is True else 'execute'
GKSystem.getSystem().executeAction(mode, replication, [], "")
