"""Contains the Flow/Aimsun API manager."""
import socket
import time
import logging
import struct

import flow.utils.aimsun.constants as ac
import flow.utils.aimsun.struct as aimsun_struct


def create_client(port, print_status=False):
    """Create a socket connection with the server.

    Parameters
    ----------
    port : int
        the port number of the socket connection
    print_status : bool, optional
        specifies whether to print a status check while waiting for connection
        between the server and client

    Returns
    -------
    socket.socket
        socket for client connection
    """
    # create a socket connection
    if print_status:
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
        data = s.recv(256)
    if print_status:
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
        self.s = create_client(port, print_status=True)

    def _send_command(self, command_type, in_format, values, out_format):
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
        in_format : str or None
            format of the input structure
        values : tuple of Any or None
            commands to be encoded and issued to the server
        out_format : str or None
            format of the output structure

        Returns
        -------
        Any
            the final message received from the Aimsun server
        """
        # send the command type to the server
        self.s.send(str(command_type).encode())

        # wait for a response
        unpacker = struct.Struct(format='i')
        data = None
        while data is None:
            data = self.s.recv(unpacker.size)

        # send the command values
        if in_format is not None:
            if in_format == 'str':
                self.s.send(str.encode(values[0]))
            else:
                packer = struct.Struct(format=in_format)
                packed_data = packer.pack(*values)
                self.s.send(packed_data)
        else:
            # if no command is needed, just send a status response
            self.s.send(str.encode('1'))

        # collect the return values
        if out_format is not None:
            if out_format == 'str':
                data = None
                while data is None:
                    data = self.s.recv(256)
                unpacked_data = data.decode('utf-8')
            else:
                unpacker = struct.Struct(format=out_format)
                data = None
                while data is None:
                    data = self.s.recv(unpacker.size)
                unpacked_data = unpacker.unpack(data)

            return unpacked_data

    def simulation_step(self):
        """Advance the simulation by one step.

        Since the connection is lost when this happens, this method also waits
        for and reconnects to the server.
        """
        self._send_command(ac.SIMULATION_STEP,
                           in_format=None, values=None, out_format=None)

        # reconnect to the server
        self.s = create_client(self.port)

    def stop_simulation(self):
        """Terminate the simulation.

        This will close the connection on both the client and server side.
        """
        # inform the simulation that it should terminate the simulation and the
        # server connection
        self._send_command(ac.SIMULATION_TERMINATE,
                           in_format=None, values=None, out_format=None)

        # terminate the connection
        self.s.close()

    def get_edge_name(self, edge):
        """Get the name of an edge in Aimsun.

        Parameters
        ----------
        edge : str
            name of the edge in Flow

        Returns
        -------
        int
            name of the edge in Aimsun
        """
        return self._send_command(ac.GET_EDGE_NAME,
                                  in_format='str',
                                  values=(edge,),
                                  out_format='i')[0]

    def add_vehicle(self, edge, lane, type_id, pos, speed, next_section):
        """Add a vehicle to the network.

        Parameters
        ----------
        edge : int
            name of the start edge
        lane : int
            start lane
        type_id : int
            vehicle type
        pos : float
            starting position
        speed : float
            starting speed
        next_section : int
            the edge number the vehicle should move towards after the current
            edge it is one. If set to -1, the vehicle takes the next feasible
            route

        Returns
        -------
        int
            name of the new vehicle in Aimsun
        """
        veh_id, = self._send_command(
            ac.ADD_VEHICLE,
            in_format='i i i f f i',
            values=(edge, lane, type_id, pos, speed, next_section),
            out_format='i')

        return veh_id

    def remove_vehicle(self, veh_id):
        """Remove a vehicle from the network.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        """
        self._send_command(ac.REMOVE_VEHICLE,
                           in_format='i',
                           values=(veh_id,),
                           out_format='i')

    def set_speed(self, veh_id, speed):
        """Set the speed of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        speed : float
            target speed
        """
        self._send_command(ac.VEH_SET_SPEED,
                           in_format='i f',
                           values=(veh_id, speed),
                           out_format='i')

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
        return self._send_command(ac.VEH_SET_LANE,
                                  in_format='i i',
                                  values=(veh_id, direction),
                                  out_format='i')

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
        return self._send_command(ac.VEH_SET_ROUTE,
                                  values=(veh_id, route))

    def set_color(self, veh_id, color):
        """Set the color of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun
        color : (int, int, int)
            red, green, blue values
        """
        r, g, b = color
        return self._send_command(ac.VEH_SET_COLOR,
                                  in_format='i i i i',
                                  values=(veh_id, r, g, b),
                                  out_format='i')

    def get_entered_ids(self):
        """Return the ids of all vehicles that entered the network."""
        veh_ids = self._send_command(ac.VEH_GET_ENTERED_IDS,
                                     in_format=None,
                                     values=None,
                                     out_format='str')

        if veh_ids == '-1':
            return []
        else:
            veh_ids = veh_ids.split(':')
            return [int(v) for v in veh_ids]

    def get_exited_ids(self):
        """Return the ids of all vehicles that exited the network."""
        veh_ids = self._send_command(ac.VEH_GET_EXITED_IDS,
                                     in_format=None,
                                     values=None,
                                     out_format='str')

        if veh_ids == '-1':
            return []
        else:
            veh_ids = veh_ids.split(':')
            return [int(v) for v in veh_ids]

    def get_vehicle_type_id(self, flow_id):
        """Get's the Aimsun type number of a Flow vehicle types.

        Parameters
        ----------
        flow_id : str
            Flow-specific vehicle type

        Returns
        -------
        int
            Aimsun-specific vehicle type
        """
        return self._send_command(ac.VEH_GET_TYPE_ID,
                                  in_format='str',
                                  values=(flow_id,),
                                  out_format='i')[0]

    def get_vehicle_static_info(self, veh_id):
        """Return the static information of the specified vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        flow.utils.aimsun.struct.StaticInfVeh
            static info object
        """
        static_info = aimsun_struct.StaticInfVeh()

        (static_info.report,
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
         static_info.idLine) = self._send_command(
            ac.VEH_GET_STATIC,
            in_format='i',
            values=(veh_id,),
            out_format='i i i f f f f f f f f f f i i i ? f f f f f i i i i')

        return static_info

    def get_vehicle_tracking_info(self, veh_id):
        """Return the tracking information of the specified vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        flow.utils.aimsun.struct.InfVeh
            tracking info object
        """
        tracking_info = aimsun_struct.InfVeh()

        (tracking_info.report,
         tracking_info.idVeh,
         tracking_info.type,
         tracking_info.CurrentPos,
         tracking_info.distance2End,
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
         tracking_info.SystemEntranceT,
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
         tracking_info.idLaneTo) = self._send_command(
            ac.VEH_GET_TRACKING,
            in_format='i',
            values=(veh_id,),
            out_format='i i i f f f f f f f f f f f f f f f f i i i i i i i i')

        return tracking_info

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
        return self._send_command(ac.VEH_GET_LEADER,
                                  in_format='i',
                                  values=(veh_id,),
                                  out_format='i')[0]

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
        return self._send_command(ac.VEH_GET_FOLLOWER,
                                  in_format='i',
                                  values=(veh_id,),
                                  out_format='i')[0]

    def get_vehicle_headway(self, veh_id):
        """Return the headway of a specific vehicle.

        Parameters
        ----------
        veh_id : int
            name of the vehicle in Aimsun

        Returns
        -------
        float
            headway
        """
        return self._send_command(ac.VEH_GET_HEADWAY,
                                  in_format='i',
                                  values=(veh_id,),
                                  out_format='f')[0]

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
        return self._send_command(ac.VEH_GET_ROUTE,
                                  values=[veh_id])

    def get_traffic_light_ids(self):
        """Return the ids of all traffic lights in the network."""
        tl_ids = self._send_command(ac.TL_GET_IDS,
                                    in_format=None,
                                    values=None,
                                    out_format='str')

        if tl_ids == '-1':
            return []
        else:
            tl_ids = tl_ids.split(':')
            return [int(t) for t in tl_ids]

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
        return self._send_command(ac.TL_GET_STATE, values=(tl_id,))

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
        """
        self._send_command(ac.TL_SET_STATE,
                           values=[tl_id, link_index, state],
                           out_format=None)
