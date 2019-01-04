"""Aimsun dummy server.

This script creates a dummy server mimicking the functionality in the Aimsun
runner script. Used for testing purposes.
"""
from thread import start_new_thread
import socket
import struct
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import flow.utils.aimsun.constants as ac  # noqa

PORT = 9999
entered_vehicles = [1, 2, 3, 4, 5]
exited_vehicles = [6, 7, 8, 9, 10]
tl_ids = [1, 2, 3, 4, 5]


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

            if data == ac.VEH_GET_ENTERED_IDS:
                send_message(conn, in_format='i', values=(0,))
                data = None
                while data is None:
                    data = conn.recv(256)
                global entered_vehicles
                if len(entered_vehicles) == 0:
                    output = '-1'
                else:
                    output = ':'.join([str(e) for e in entered_vehicles])
                conn.send(output)
                entered_vehicles = []

            elif data == ac.VEH_GET_EXITED_IDS:
                send_message(conn, in_format='i', values=(0,))
                data = None
                while data is None:
                    data = conn.recv(256)
                global exited_vehicles
                if len(exited_vehicles) == 0:
                    output = '-1'
                else:
                    output = ':'.join([str(e) for e in exited_vehicles])
                conn.send(output)
                exited_vehicles = []

            elif data == ac.VEH_GET_STATIC:
                send_message(conn, in_format='i', values=(0,))
                retrieve_message(conn, 'i')
                output = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, False, 18, 19, 20, 21, 22, 23, 24, 25, 26)
                send_message(conn,
                             in_format='i i i f f f f f f f f f f i i i ? '
                                       'f f f f f i i i i',
                             values=output)

            elif data == ac.VEH_GET_TRACKING:
                send_message(conn, in_format='i', values=(0,))
                retrieve_message(conn, 'i')
                output = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
                send_message(conn,
                             in_format='i i i f f f f f f f f f f f f f f f f '
                                       'i i i i i i i i',
                             values=output)

            elif data == ac.TL_GET_IDS:
                send_message(conn, in_format='i', values=(0,))
                data = None
                while data is None:
                    data = conn.recv(256)
                global tl_ids
                if len(tl_ids) == 0:
                    output = '-1'
                else:
                    output = ':'.join([str(e) for e in tl_ids])
                conn.send(output)
                tl_ids = []

            # in case the message is unknown, return -1001
            else:
                send_message(conn, in_format='i', values=(-1001,))


while True:
    # tcp/ip connection from the aimsun process
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', PORT))

    # connect to the Flow instance
    server_socket.listen(10)
    c, address = server_socket.accept()

    # start the threaded process
    start_new_thread(threaded_client, (c,))
