import flow.config as config
import sys
import os

sys.path.append(os.path.join(config.AIMSUN_NEXT_PATH,
                             'programming/Aimsun Next API/AAPIPython/Micro'))

import flow.utils.aimsun.constants as ac
import AAPI as aimsun_api
from AAPI import *
from PyANGKernel import *
import socket
import struct
from thread import start_new_thread
import numpy as np

PORT = 9999
entered_vehicles = []
exited_vehicles = []


def send_message(conn, in_format, values):
    """Send a message to the client.

    Parameters
    ----------
    conn : socket.socket
        socket for server connection
    in_format : str
        format of the input structure
    values : tuple of Any
        commands to be encoded and issued to the client
    """
    packer = struct.Struct(format=in_format)
    packed_data = packer.pack(*values)
    conn.send(packed_data)


def retrieve_message(conn, out_format):
    """Retrieve a message from the client.

    Parameters
    ----------
    conn : socket.socket
        socket for server connection
    out_format : str or None
        format of the output structure

    Returns
    -------
    Any
        received message
    """
    unpacker = struct.Struct(format=out_format)
    try:
        data = conn.recv(unpacker.size)
        unpacked_data = unpacker.unpack(data)
    finally:
        pass
    return unpacked_data


def threaded_client(conn):
    # send feedback that the connection is active
    conn.send('Ready.')

    done = False
    while not done:
        # receive the next message
        data = conn.recv(256)

        if data is not None:
            # if the message is empty, search for the next message
            if data == '':
                continue

            # convert to integer
            data = int(data)

            # if the simulation step is over, terminate the loop and let
            # the step be executed
            if data == ac.SIMULATION_STEP:
                send_message(conn, in_format='i', values=(0,))
                done = True

            # Note that alongside this, the process is closed in Flow,
            # thereby terminating the socket connection as well.
            elif data == ac.SIMULATION_TERMINATE:
                send_message(conn, in_format='i', values=(0,))
                done = True

            elif data == ac.ADD_VEHICLE:
                send_message(conn, in_format='i', values=(0,))

                edge, lane, type_id, pos, speed, next_section = \
                    retrieve_message(conn, 'i i i f f i')

                # 1 if tracked, 0 otherwise
                tracking = 1

                veh_id = aimsun_api.AKIPutVehTrafficFlow(
                    edge, lane+1, type_id, pos, speed, next_section,
                    tracking
                )

                send_message(conn, in_format='i', values=(veh_id,))

            elif data == ac.REMOVE_VEHICLE:
                send_message(conn, in_format='i', values=(0,))
                veh_id, = retrieve_message(conn, 'i')
                aimsun_api.AKIVehTrackedRemove(veh_id)
                send_message(conn, in_format='i', values=(0,))

            elif data == ac.VEH_SET_SPEED:
                send_message(conn, in_format='i', values=(0,))
                veh_id, speed = retrieve_message(conn, 'i f')
                new_speed = speed * 3.6
                #aimsun_api.AKIVehTrackedForceSpeed(veh_id, new_speed)
                aimsun_api.AKIVehTrackedModifySpeed(veh_id, new_speed)
                send_message(conn, in_format='i', values=(0,))

            elif data == ac.VEH_SET_LANE:
                conn.send('Set vehicle lane.')
                veh_id, target_lane = retrieve_message(conn, 'i i')
                aimsun_api.AKIVehTrackedModifyLane(veh_id, target_lane)
                send_message(conn, in_format='i', values=(0,))

            elif data == ac.VEH_SET_ROUTE:
                send_message(conn, in_format='i', values=(0,))
                # TODO

            elif data == ac.VEH_SET_COLOR:
                send_message(conn, in_format='i', values=(0,))
                veh_id, r, g, b = retrieve_message(conn, 'i i i i')
                # TODO
                send_message(conn, in_format='i', values=(0,))

            elif data == ac.VEH_GET_ENTERED_IDS:
                send_message(conn, in_format='i', values=(0,))

                # wait for a response
                data = None
                while data is None:
                    data = conn.recv(256)

                # send the entered vehicle IDs
                global entered_vehicles

                if len(entered_vehicles) == 0:
                    output = '-1'
                else:
                    output = ':'.join([str(e) for e in entered_vehicles])
                conn.send(output)

                # clear the list
                entered_vehicles = []

            elif data == ac.VEH_GET_EXITED_IDS:
                send_message(conn, in_format='i', values=(0,))

                # wait for a response
                data = None
                while data is None:
                    data = conn.recv(256)

                # send the entered vehicle IDs
                global exited_vehicles

                if len(exited_vehicles) == 0:
                    output = '-1'
                else:
                    output = ':'.join([str(e) for e in exited_vehicles])
                conn.send(output)

                # clear the list
                exited_vehicles = []

            elif data == ac.VEH_GET_TYPE_ID:
                send_message(conn, in_format='i', values=(0,))

                # get the type ID in flow
                type_id = None
                while type_id is None:
                    type_id = conn.recv(256)

                # convert the edge name to an edge name in Aimsun
                model = GKSystem.getSystem().getActiveModel()
                type_vehicle = model.getType("GKVehicle")
                vehicle = model.getCatalog().findByName(
                    type_id, type_vehicle)
                aimsun_type = vehicle.getId()
                aimsun_type_pos = AKIVehGetVehTypeInternalPosition(aimsun_type)

                send_message(conn, in_format='i', values=(aimsun_type_pos,))

            elif data == ac.VEH_GET_STATIC:
                send_message(conn, in_format='i', values=(0,))
                veh_id, = retrieve_message(conn, 'i')

                static_info = aimsun_api.AKIVehGetStaticInf(veh_id)
                output = (static_info.report,
                          static_info.idVeh,
                          static_info.type,
                          static_info.length,
                          static_info.width,
                          static_info.maxDesiredSpeed,
                          static_info.maxAcceleration,
                          static_info.normalDeceleration,
                          static_info.maxDeceleration,
                          static_info.speedAcceptance,
                          static_info.minDistanceVeh,
                          static_info.giveWayTime,
                          static_info.guidanceAcceptance,
                          static_info.enrouted,
                          static_info.equipped,
                          static_info.tracked,
                          static_info.keepfastLane,
                          static_info.headwayMin,
                          static_info.sensitivityFactor,
                          static_info.reactionTime,
                          static_info.reactionTimeAtStop,
                          static_info.reactionTimeAtTrafficLight,
                          static_info.centroidOrigin,
                          static_info.centroidDest,
                          static_info.idsectionExit,
                          static_info.idLine)

                send_message(conn,
                             in_format='i i i f f f f f f f f f f i i i ? '
                                       'f f f f f i i i i',
                             values=output)

            elif data == ac.VEH_GET_TRACKING:
                send_message(conn, in_format='i', values=(0,))

                veh_id, = retrieve_message(conn, 'i')

                tracking_info = aimsun_api.AKIVehTrackedGetInf(veh_id)
                output = (
                          # tracking_info.report,
                          # tracking_info.idVeh,
                          # tracking_info.type,
                          tracking_info.CurrentPos,
                          tracking_info.distance2End,
                          tracking_info.xCurrentPos,
                          tracking_info.yCurrentPos,
                          tracking_info.zCurrentPos,
                          # tracking_info.xCurrentPosBack,
                          # tracking_info.yCurrentPosBack,
                          # tracking_info.zCurrentPosBack,
                          tracking_info.CurrentSpeed,
                          # tracking_info.PreviousSpeed,
                          tracking_info.TotalDistance,
                          # tracking_info.SystemGenerationT,
                          # tracking_info.SystemEntranceT,
                          tracking_info.SectionEntranceT,
                          tracking_info.CurrentStopTime,
                          tracking_info.stopped,
                          tracking_info.idSection,
                          tracking_info.segment,
                          tracking_info.numberLane,
                          tracking_info.idJunction,
                          tracking_info.idSectionFrom,
                          tracking_info.idLaneFrom,
                          tracking_info.idSectionTo,
                          tracking_info.idLaneTo)

                send_message(conn,
                             in_format='f f f f f f f f f f i i i i i i i i',
                             values=output)

            elif data == ac.VEH_GET_LEADER:
                send_message(conn, in_format='i', values=(0,))
                veh_id, = retrieve_message(conn, 'i')
                leader = aimsun_api.AKIVehGetLeaderId(veh_id)
                send_message(conn, in_format='i', values=(leader,))

            elif data == ac.VEH_GET_FOLLOWER:
                send_message(conn, in_format='i', values=(0,))
                veh_id, = retrieve_message(conn, 'i')
                follower = aimsun_api.AKIVehGetFollowerId(veh_id)
                send_message(conn, in_format='i', values=(follower,))

            elif data == ac.VEH_GET_NEXT_SECTION:
                send_message(conn, in_format='i', values=(0,))
                veh_id, section = retrieve_message(conn, 'i i')
                next_section = AKIVehInfPathGetNextSection(veh_id, section)
                send_message(conn, in_format='i', values=(next_section,))

            elif data == ac.VEH_GET_ROUTE:
                send_message(conn, in_format='i', values=(0,))
                veh_id, = retrieve_message(conn, 'i')
                # TODO

            elif data == ac.TL_GET_IDS:
                send_message(conn, in_format='i', values=(0,))
                # TODO

            elif data == ac.TL_SET_STATE:
                send_message(conn, in_format='i', values=(0,))
                # TODO

            elif data == ac.TL_GET_STATE:
                send_message(conn, in_format='i', values=(0,))
                tl_id, = retrieve_message(conn, 'i')
                # TODO

            elif data == ac.GET_EDGE_NAME:
                send_message(conn, in_format='i', values=(0,))

                # get the edge ID in flow
                edge = None
                while edge is None:
                    edge = conn.recv(256)

                model = GKSystem.getSystem().getActiveModel()
                edge_aimsun = model.getCatalog().findByName(
                    edge, model.getType('GKSection'))

                send_message(conn, in_format='i', values=(edge_aimsun.getId(),))

            # in case the message is unknown, return -1001
            else:
                send_message(conn, in_format='i', values=(-1001,))

    # close the connection
    conn.close()


def AAPILoad():
    return 0


def AAPIInit():
    # set the simulation time to be very large
    AKISetEndSimTime(2e6)
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # tcp/ip connection from the aimsun process
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', PORT))

    # connect to the Flow instance
    server_socket.listen(10)
    c, address = server_socket.accept()

    # start the threaded process
    start_new_thread(threaded_client, (c,))

    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    return 0


def AAPIFinish():
    return 0


def AAPIUnLoad():
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    return 0


def AAPIEnterVehicle(idveh, idsection):
    global entered_vehicles
    entered_vehicles.append(idveh)
    return 0


def AAPIExitVehicle(idveh, idsection):
    global exited_vehicles
    exited_vehicles.append(idveh)
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
