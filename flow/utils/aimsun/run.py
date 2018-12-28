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
        data = conn.recv(2048)

        if data is not None:
            # if the message is empty, search for the next message
            if data == '':
                continue

            # convert to integer
            data = int(data)

            # if the simulation step is over, terminate the loop and let
            # the step be executed
            if data == ac.SIMULATION_STEP:
                conn.send('Simulation step.')
                done = True

                # TODO: is this right?
                # order = 3  # 3: Stop simulation
                # when = int(aimsun_api.AKIGetCurrentSimulationTime())
                # aimsun_api.ANGSetSimulationOrder(order, when)
                #
                # order = 0
                # when = int(when + self.sim_step)
                # # TODO this is risky since "when" should be integer
                # aimsun_api.ANGSetSimulationOrder(order, when)

            # Note that alongside this, the process is closed in Flow,
            # thereby terminating the socket connection as well.
            elif data == ac.SIMULATION_TERMINATE:
                conn.send('Terminate simulation.')

                # # Stop simulation
                # order = 1  # 1: Cancel simulation
                # when = int(aimsun_api.AKIGetCurrentSimulationTime())
                # aimsun_api.ANGSetSimulationOrder(order, when)
                #
                # # Save and close the network
                # gui = GKGUISystem.getGUISystem().getActiveGui()
                # model = gui.getActiveModel()
                # # TODO this can be saveAs depending on the template
                # gui.save()
                # gui.closeDocument(model)
                #
                # # Quit the GUI
                # gui.forceQuit()

            elif data == ac.ADD_VEHICLE:
                conn.send('Add vehicle')

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
                conn.send('Remove vehicle')
                veh_id, = retrieve_message(conn, 'i')
                aimsun_api.AKIVehTrackedRemove(veh_id)

            elif data == ac.VEH_SET_SPEED:
                conn.send('Set vehicle speed.')
                veh_id, speed = retrieve_message(conn, 'i f')
                aimsun_api.AKIVehTrackedModifySpeed(veh_id, speed)

            elif data == ac.VEH_SET_LANE:
                conn.send('Set vehicle lane.')
                veh_id, target_lane = retrieve_message(conn, 'i i')
                aimsun_api.AKIVehTrackedModifyLane(veh_id, target_lane)

            elif data == ac.VEH_SET_ROUTE:
                conn.send('Set vehicle route.')
                # TODO

            elif data == ac.VEH_SET_COLOR:
                conn.send('Set vehicle color.')
                veh_id, r, g, b = retrieve_message(conn, 'i i i i')
                # TODO

            elif data == ac.VEH_GET_TYPE_ID:
                conn.send('Get vehicle type.')

                # get the type ID in flow
                type_id = None
                while type_id is None:
                    type_id = conn.recv(2048)

                # convert the edge name to an edge name in Aimsun
                model = GKSystem.getSystem().getActiveModel()
                type_vehicle = model.getType("GKVehicle")
                vehicle = model.getCatalog().findByName(
                    type_id, type_vehicle)
                aimsun_type = vehicle.getId()
                aimsun_type_pos = AKIVehGetVehTypeInternalPosition(aimsun_type)

                send_message(conn, in_format='i', values=(aimsun_type_pos,))

            elif data == ac.VEH_GET_STATIC:
                conn.send('Get vehicle static info.')
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
                conn.send('Get vehicle tracking info.')

                veh_id, = retrieve_message(conn, 'i')

                tracking_info = aimsun_api.AKIVehTrackedGetInf(veh_id)
                output = (tracking_info.report,
                          tracking_info.idVeh,
                          tracking_info.type,
                          tracking_info.CurrentPos,
                          tracking_info.distnace2End,
                          tracking_info.xCurrentPos,
                          tracking_info.yCurrentPos,
                          tracking_info.zCurrentPos,
                          tracking_info.xCurrentPosBack,
                          tracking_info.yCurrentPosBack,
                          tracking_info.zCurrentPosBack,
                          tracking_info.CurrentSpeed,
                          tracking_info.PreviousSpeed,
                          tracking_info.TotalDistance,
                          tracking_info.SystemGenerationT,
                          tracking_info.SectionEntranceT,
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
                             in_format='i i i f f f f f f f f f f f f f f '
                                       'f f i i i i i i i i',
                             values=output)

            elif data == ac.VEH_GET_LEADER:
                conn.send('Get vehicle leader.')
                veh_id, = retrieve_message(conn, 'i')
                leader = aimsun_api.AKIVehGetLeaderId(veh_id)
                send_message(conn, in_format='i', values=(leader,))

            elif data == ac.VEH_GET_FOLLOWER:
                conn.send('Get vehicle follower.')
                veh_id, = retrieve_message(conn, 'i')
                follower = aimsun_api.AKIVehGetFollowerId(veh_id)
                send_message(conn, in_format='i', values=(follower,))

            elif data == ac.VEH_GET_ROUTE:
                conn.send('Get vehicle route.')
                veh_id, = retrieve_message(conn, 'i')
                # TODO

            elif data == ac.TL_GET_IDS:
                conn.send('Get traffic light ids')
                # TODO

            elif data == ac.TL_SET_STATE:
                conn.send('Set traffic light state')
                # TODO

            elif data == ac.TL_GET_STATE:
                conn.send('Get traffic light state')
                tl_id, = retrieve_message(conn, 'i')
                # TODO

            elif data == ac.GET_EDGE_NAME:
                conn.send('Get edge name.')

                # get the edge ID in flow
                edge = None
                while edge is None:
                    edge = conn.recv(2048)

                model = GKSystem.getSystem().getActiveModel()
                edge_aimsun = model.getCatalog().findByName(
                    edge, model.getType('GKSection'))

                send_message(conn, in_format='i', values=(edge_aimsun.getId(),))

            # in case the message is an error message or unknown
            else:
                conn.send('Failure.')

    # close the connection
    conn.close()


def AAPILoad():
    return 0


def AAPIInit():
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # # tcp/ip connection from the aimsun process
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server_socket.bind(('localhost', PORT))
    #
    # # connect to the Flow instance
    # server_socket.listen(10)
    # c, address = server_socket.accept()
    #
    # # start the threaded process
    # start_new_thread(threaded_client, (c,))

    type_id = 'idm'
    model = GKSystem.getSystem().getActiveModel()
    type_vehicle = model.getType("GKVehicle")
    vehicle = model.getCatalog().findByName(
        type_id, type_vehicle)
    aimsun_type = vehicle.getId()

    idVeh = ANGConnGetObjectIdByType(AKIConvertFromAsciiString("Car"), None,
                                     False)
    vehPos = AKIVehGetVehTypeInternalPosition(idVeh)
    print ("veh pos", vehPos)
    aimsun_type_pos = AKIVehGetVehTypeInternalPosition(aimsun_type)

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
