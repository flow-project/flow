import sys
sys.path.append('/home/aboudy/Aimsun_Next_8_3_0/programming/Aimsun Next API/AAPIPython/Micro')

import AAPI as aimsun_api
import socket
from thread import start_new_thread

PORT = 9999


def AAPILoad():
    aimsun_api.AKIPrintString("AAPILoad")
    return 0


def AAPIInit():
    aimsun_api.AKIPrintString("AAPIInit")
    # TODO; trigger the environment
    return 0


def AAPIManage(time, timeSta, timeTrans, acycle):
    # connect as a client to the server created by Flow
    aimsun_api.AKIPrintString("Listening")
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connected = False
    # num_tries = 0
    # while not connected and num_tries < 100:
    #     num_tries += 1
    #     try:
    #         s.connect(('localhost', PORT))
    #         connected = True
    #     except Exception as e:
    #         aimsun_api.AKIPrintString(
    #             "Cannot connect to the server: {}".format(e))
    # aimsun_api.AKIPrintString("Connected")

    # tcp/ip connection with the aimsun process
    HOST = 'localhost'
    PORT = 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))

    # connect to an instance to receive initial state information
    server_socket.listen(10)
    print("Listening")
    conn, address = server_socket.accept()
    print("Connected")

    def threaded_client(conn):
        conn.send("Welcome! You are connected!")

        while True:
            data = conn.recv(2048)
            if data:
                AKIPutVehTrafficFlow(edge_aimsun_id, lane, type_id, pos, speed,
                                     next_section, tracking)
                exit()
            if not data:
                break

        # close the connection
        conn.close()

    start_new_thread(threaded_client, (conn,))

    # done = False
    # while not done:
    #     aimsun_api.AKIPrintString("woop")
    #     # read tcp/ip commands from the Flow server
    #     message = s.recv(1024)
    #     aimsun_api.AKIPrintString("woop")
    #
    #     # once a message is received
    #     if message is not None:
    #         aimsun_api.AKIPrintString("woop")
    #         aimsun_api.AKIPrintString(message)
    #         message_type = message[0]
    #         # check if it is an add vehicle message
    #         # if message_type == ac.ADD_VEHICLE:
    #         #     pass

    #             edge_aimsun_id, lane, type_id, pos, speed, next_section, tracking = message_content
    #             # add vehicle in Aimsun
    #             next_section = -1  # negative one means the first feasible turn #TODO get route
    #             tracking = 1  # 1 if tracked, 0 otherwise
    #             type_id = 1
    #             id = AKIPutVehTrafficFlow(edge_aimsun_id, lane, type_id, pos, speed, next_section, tracking)

    return 0


def AAPIPostManage(time, timeSta, timeTrans, acycle):
    aimsun_api.AKIPrintString("AAPIPostManage")
    return 0


def AAPIFinish():
    aimsun_api.AKIPrintString("AAPIFinish")
    return 0


def AAPIUnLoad():
    # AKIPrintString("AAPIUnLoad")
    return 0


def AAPIPreRouteChoiceCalculation(time, timeSta):
    # AKIPrintString("AAPIPreRouteChoiceCalculation")
    return 0


def AAPIEnterVehicle(idveh, idsection):
    return 0


def AAPIExitVehicle(idveh, idsection):
    return 0


def AAPIEnterPedestrian(idPedestrian, originCentroid):
    return 0


def AAPIExitPedestrian(idPedestrian, destinationCentroid):
    return 0


def AAPIEnterVehicleSection(idveh, idsection, atime):
    return 0


def AAPIExitVehicleSection(idveh, idsection, atime):
    return 0
