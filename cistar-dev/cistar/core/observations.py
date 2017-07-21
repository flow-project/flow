"""
This script contains a series of observation states that can be used while learning traffic scenarios.
"""

import numpy as np
from numpy.random import normal

"""
The simple network observation spaces can be used for networks that can be treated as if they are part of
a single continuous edge. Networks of this type include the ring road, figure8, and two-way intersections.
-------
Use single_lane versions if the edges of the network contain only one lane.
"""


def simple_network_global(vehicle_data, id_sequence=None, multiple_lanes=False, **kwargs):
    """
    Observation space that can be used for networks that can be treated as if they are part of a single
    continuous edge. Networks of this type include the ring road, figure8, and two-way intersections.
    -------
    The observation outputted by this function provides relevant information on all vehicles in the network.

    :param vehicle_data: a dict depicting the properties of the vehicles at any given time step
    :param id_sequence: sequence of vehicle ids depicting how the data should be ordered
                        If no sequence is given, the data is ordered as it is in the vehicle_data dict
    :param multiple_lanes: bool value depicting whether the edges of the network contain one or multiple lanes
    :param kwargs: additional commands, which may include the intensity of observation (speed and pos) noise
    :return: a matrix of vehicle speeds and absolute positions, ordered according to the id sequence
    """
    # if no sequence is given, the data is ordered as it is in the vehicle_data dict
    if id_sequence is None:
        id_sequence = list(vehicle_data.keys())

    # if the intensity of gaussian noise on either the position or velocity observations
    # is not provided, it is set to zero
    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    if multiple_lanes:
        observation = np.array([[vehicle_data[veh_id]["speed"] + normal(0, kwargs["observation_vel_std"]),
                                 vehicle_data[veh_id]["absolute_position"] + normal(0, kwargs["observation_pos_std"]),
                                 vehicle_data[veh_id]["lane"]] for veh_id in id_sequence]).T
    else:
        observation = np.array([[vehicle_data[veh_id]["speed"] + normal(0, kwargs["observation_vel_std"]),
                                 vehicle_data[veh_id]["absolute_position"] + normal(0, kwargs["observation_pos_std"])]
                                for veh_id in id_sequence]).T

    return observation


def simple_network_local(vehicle_data, veh_id, multiple_lanes=False, **kwargs):
    """
    Observation space that can be used for networks that can be treated as if they are part of a single
    continuous edge. Networks of this type include the ring road, figure8, and two-way intersections.
    -------
    The observation outputted by this function provides relevant information on the specified vehicle and all
    vehicles adjacent to it in the network.

    :param vehicle_data: a dict depicting the properties of the vehicles at any given time step
    :param veh_id: unique vehicle identifier
    :param multiple_lanes:
    :param kwargs:
    :return:
    """
    # if the intensity of gaussian noise on either the position or velocity observations
    # is not provided, it is set to zero
    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    # the type is used to specify which adjacent vehicles are to be added to the observation space
    # available choices are: "front", "back", and "both"; default is "both"
    if "type" not in kwargs:
        kwargs["type"] = "both"

    trail_id = vehicle_data[veh_id]["follower"]
    lead_id = vehicle_data[veh_id]["leader"]

    # create the id_sequence to be used in the observation space
    # the id_sequence consists of the the specified veh_id, followed by its leaders and/or followers, in that order
    id_sequence = [veh_id]
    leading_id_sequence = []
    trailing_id_sequence = []
    if kwargs["type"] == "front" or kwargs["type"] == "both":
        if type(lead_id) == list:
            for vehID in lead_id:
                id_sequence.append(vehID)
                leading_id_sequence.append(vehID)
        else:
            id_sequence.append(lead_id)
            leading_id_sequence.append(lead_id)

    if kwargs["type"] == "back" or kwargs["type"] == "both":
        if type(trail_id) == list:
            for vehID in lead_id:
                id_sequence.append(vehID)
                trailing_id_sequence.append(vehID)
        else:
            id_sequence.append(trail_id)
            trailing_id_sequence.append(trail_id)

    veh_observation = np.array([[vehicle_data[veh_id]["speed"]],
                                [vehicle_data[veh_id]["absolute_position"]]])

    this_pos = vehicle_data[veh_id]["absolute_pos"]
    this_length = vehicle_data[veh_id]["length"]
    leading_pos = np.array([vehicle_data[vehID]["absolute_position"] for vehID in leading_id_sequence])
    leading_length = np.array([vehicle_data[vehID]["length"] for vehID in leading_id_sequence])
    trailing_pos = np.array([vehicle_data[vehID]["absolute_position"] for vehID in trailing_id_sequence])
    headway = np.append(np.mod(leading_pos - leading_length - this_pos),
                        np.array([]))

    adj_observation = np.array([[vehicle_data[vehID]["speed"], headway[i]]
                                for i, vehID in enumerate(id_sequence)]).T

    return np.hstack((veh_observation, adj_observation))


def complex_network_global(vehicle_data, id_sequence=None, **kwargs):
    """

    :param vehicle_data: a dict depicting the properties of the vehicles at any given time step
    :param id_sequence:
    :param kwargs:
    :return:
    """
    # if no sequence is given, the data is ordered as it is in the vehicle_data dict
    if id_sequence is None:
        id_sequence = list(vehicle_data.keys())

    # if the intensity of gaussian noise on either the position or velocity observations
    # is not provided, it is set to zero
    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0


def complex_network_local(vehicle_data, veh_id, **kwargs):
    """

    :param vehicle_data:
    :param veh_id:
    :param kwargs:
    :return:
    """
    # if the intensity of gaussian noise on either the position or velocity observations
    # is not provided, it is set to zero
    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0

    if "observation_vel_std" not in kwargs:
        kwargs["observation_vel_std"] = 0


def loop_detector_global(loop_detector_data, id_sequence, **kwargs):
    """

    :param loop_detector_data:
    :param id_sequence:
    :param kwargs:
    :return:
    """
    pass


def loop_detector_local(loop_detector_data, edge_id, **kwargs):
    """

    :param loop_detector_data:
    :param edge_id:
    :param kwargs:
    :return:
    """
    pass
