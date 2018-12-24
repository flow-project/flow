"""Contains the Flow/Aimsun API manager."""
import socket
import time
import logging

import flow.utils.aimsun.constants as ac


def create_client(port):
    """Create a socket connection with the server.

    Parameters
    ----------
    port : int
        the port number of the socket connection

    Returns
    -------
    FIXME
        socket for client connection
    """
    # create a socket connection
    print('Listening for connection...', end=' ')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    num_tries = 0
    while not connected and num_tries < 100:
        num_tries += 1
        try:
            s.connect(('localhost', port))
            connected = True
        except Exception as e:
            logging.debug('Cannot connect to the server: {}'.format(e))
            time.sleep(1)

    # check the connection
    data = None
    while data is None:
        data = s.recv(2048)
    print(data.decode('utf-8'))

    return s


class FlowAimsunAPI(object):
    """An API used to interact with Aimsun via a TCP connection.

    This is needed since Aimsun is written in Python 2.7.4, and may be
    deprecated in the future. An server/client connection is created between
    Flow and the Aimsun run script. The client is passed to this object and
    commands are accordingly provided to the Aimsun sever via this client.
    """

    def __init__(self, port):
        """Instantiate the API.

        Parameters
        ----------
        port : int
            the port number of the socket connection
        """
        self.port = port
        self.s = create_client(port)

    def _send_command(self, command_type, values):  # TODO: return_type
        """Send an arbitrary command via the connection.

        Commands are sent in two stages. First, the client sends the command
        type (e.g. ac.REMOVE_VEHICLE) and waits for a conformation message from
        the server. Once the confirmation is received, the client send a
        encoded binary packet that the server will be prepared to decode, and
        will then receive some return value (either the value the client was
        requesting or a 0 signifying that the command has been executed. This
        value is then returned by this method.

        Parameters
        ----------
        command_type : flow.utils.aimsun.constants.*
            the command the client would like Aimsun to execute
        values : list of Any
            list of commands to be encoded and issued to the server

        Returns
        -------
        Any
            the final message received from the Aimsun server
        """
        # send the command type to the server
        self.s.send(str(command_type).encode())

        # wait for a response
        data = None
        while data is None:
            data = self.s.recv(2048)
        print(data.decode('utf-8'))

    def simulation_step(self):
        """Advance the simulation by one step.

        Since the connection is lost when this happens, this method also waits
        for and reconnects to the server.
        """
        self._send_command(ac.SIMULATION_STEP, values=[])

        # reconnect to the server
        self.s = create_client(self.port)

    def stop_simulation(self):
        """Terminate the simulation.

        This will close the connection on both the client and server side.
        """
        # inform the simulation that it should terminate the simulation and the
        # server connection
        self._send_command(ac.SIMULATION_TERMINATE, values=[])

        # terminate the connection
        self.s.close()

    def add_vehicle(self, edge, lane, type_id, pos, speed, next_section):
        """

        :param veh_id:
        :return:
        """
        self._send_command(
            ac.ADD_VEHICLE,
            values=[edge, lane, type_id, pos, speed, next_section])

    def remove_vehicle(self, veh_id):
        """Remove a vehicle from the network.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        float
            status (should be 0)
        """
        self._send_command(ac.REMOVE_VEHICLE, values=[veh_id])

    def set_speed(self, veh_id, speed):
        """Set the speed of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        speed : float
            target speed

        Returns
        -------
        float
            status (should be 0)
        """
        return self._send_command(ac.VEH_SET_SPEED, values=[veh_id, speed])

    def apply_lane_change(self, veh_id, direction):
        """Set the lane change action of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        direction : int
            target direction

        Returns
        -------
        float
            status (should be 0)
        """
        return self._send_command(ac.VEH_SET_LANE, values=[veh_id, direction])

    def set_route(self, veh_id, route):
        """Set the route of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        route : list of int
            list of edges the vehicle should traverse

        Returns
        -------
        float
            status (should be 0)
        """
        return self._send_command(ac.VEH_SET_ROUTE, values=[veh_id, route])

    def set_color(self, veh_id, color):
        """Set the color of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        color : (int, int, int)
            red, green, blue values

        Returns
        -------
        float
            status (should be 0)
        """
        return self._send_command(ac.VEH_SET_COLOR, values=[veh_id, color])

    def get_vehicle_ids(self):
        """Return the ids of all vehicles in the network."""
        return self._send_command(ac.VEH_GET_IDS, values=[])

    def get_vehicle_static_info(self, veh_id):
        """

        :param veh_id:
        :return:
        """
        return self._send_command(ac.VEH_GET_STATIC, values=[veh_id])

    def get_vehicle_tracking_info(self, veh_id):
        """

        :param veh_id:
        :return:
        """
        return self._send_command(ac.VEH_GET_TRACKING, values=[veh_id])

    def get_vehicle_leader(self, veh_id):
        """Return the leader of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        int
            name of the leader
        """
        return self._send_command(ac.VEH_GET_LEADER, values=[veh_id])

    def get_vehicle_follower(self, veh_id):
        """Return the follower of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        int
            name of the follower
        """
        return self._send_command(ac.VEH_GET_FOLLOWER, values=[veh_id])

    def get_route(self, veh_id):
        """Return the route of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        list of int
            list of edge names in Aimsun
        """
        return self._send_command(ac.VEH_GET_ROUTE, values=[veh_id])

    def get_traffic_light_ids(self):
        """Return the ids of all traffic lights in the network."""
        return self._send_command(ac.TL_GET_IDS, values=[])

    def get_traffic_light_state(self, tl_id):
        """Get the traffic light state of a specific set of traffic light(s).

        Parameters
        ----------
        tl_id : int
            name of the traffic light node in Aimsun

        Returns
        -------
        str
            traffic light state of each light on that node
        """
        return self._send_command(ac.TL_GET_STATE, values=[tl_id])

    def set_traffic_light_state(self, tl_id, link_index, state):
        """Set the state of the specified traffic light(s).

        Parameters
        ----------
        tl_id : int
            name of the traffic light node in Aimsun
        link_index : TODO
            TODO
        state : str
            TODO

        Returns
        -------
        float
            status (should be 0)
        """
        return self._send_command(ac.TL_SET_STATE,
                                  values=[tl_id, link_index, state])
