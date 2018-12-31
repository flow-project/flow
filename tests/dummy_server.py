"""Aimsun dummy server.

This script creates a dummy server mimicking the functionality in the Aimsun
runner script. Used for testing purposes.
"""
import flow.utils.aimsun.constants as ac
from flow.utils.aimsun.run import send_message, retrieve_message


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

            if data == ac.VEH_GET_STATIC:
                send_message(conn, in_format='i', values=(0,))
                _ = retrieve_message(conn, 'i')
                output = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, False, 18, 19, 20, 21, 22, 23, 24, 25, 26)
                send_message(conn,
                             in_format='i i i f f f f f f f f f f i i i ? '
                                       'f f f f f i i i i',
                             values=output)

            elif data == ac.VEH_GET_TRACKING:
                send_message(conn, in_format='i', values=(0,))
                _ = retrieve_message(conn, 'i')
                output = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
                send_message(conn,
                             in_format='i i i f f f f f f f f f f f f f f f f '
                                       'i i i i i i i i',
                             values=output)

            # in case the message is unknown, return -1001
            else:
                send_message(conn, in_format='i', values=(-1001,))

    # close the connection
    conn.close()
