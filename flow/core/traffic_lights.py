import traci.constants as tc


class TrafficLights:

    def __init__(self):
        """
        Base traffic light class used to place traffic lights in the network
        and describe the state of these traffic lights. In addition, this class
        supports modifying the states of certain lights via TraCI.
        """
        # contains current time step traffic light data
        self.__tls = dict()

        # names of edges with traffic lights
        self.__ids = list()

        # traffic light properties (needed to generate proper xml files)
        self.__tls_properties = dict()

    def add(self,
            edge_id,
            tls_type="static",
            programID=None,
            offset=None,
            phases=None):
        """
        Adds a traffic light component to the network. When generating networks
        using xml files, this will explicitly place the traffic light in the
        requested edge.

        Parameters
        ----------
        edge_id : str
            name of the edge to start with traffic lights
        tls_type : str, optional
            type of the traffic light (see Note)
        programID : str, optional
            id of the traffic light program (see Note)
        offset : int, optional
            initial time offset of the program
        phases : list <dict>, optional
            list of phases to be followed by the traffic light, defaults
            to default sumo traffic light behavior. Each element in the list
            must consist of a dict with two keys:
            - "duration": length of the current phase cycle (in sec)
            - "state": string consist the sequence of states in the phase

        Note
        ----
        For information on defining traffic light properties, see:
        http://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
        """
        # add the edge id to the list of controlled edges
        self.__ids.append(edge_id)

        # prepare the data needed to generate xml files
        self.__tls_properties[edge_id] = {"id": edge_id, "type": tls_type}

        if programID is not None:
            self.__tls_properties[edge_id]["programID"] = programID

        if offset is not None:
            self.__tls_properties[edge_id]["offset"] = offset

        if phases is not None:
            self.__tls_properties[edge_id]["phase"] = phases

    def update(self, tls_subscriptions):
        """
        Updates the states and phases of the traffic lights to match current
        traffic light data.

        Parameters
        ----------
        tls_subscriptions : dict
            sumo traffic light subscription data
        """
        self.__tls = tls_subscriptions.copy()  # TODO: fix this on the base_env side

    def get_ids(self):
        """
        Returns the names of all edges with traffic lights.
        """
        return self.__ids

    def get_properties(self):
        """
        Returns traffic light properties. Meant to be used by the generator to
        import traffic light data to the .net.xml file
        """
        return self.__tls_properties

    def set_state(self, edge_id, state, env, lane_index="all"):
        """
        Sets the state of the traffic lights on a specific edge.

        Parameters
        ----------
        edge_id : str
            name of the edge with the controlled traffic lights
        state : str
            requested state for the traffic light
        env : flow.envs.base_env.Env type
            the environment at the current time step
        lane_index : int, optional
            index of the lane whose traffic light state is meant to be changed.
            If no value is provided, the lights on all lanes are updated.

        Raises
        ------
        ValueError : If the edge does not contain traffic lights.
        """
        if edge_id not in self.__ids:
            raise ValueError("Edge {} does not contain traffic lights".
                             format(edge_id))

        if lane_index == "all":
            # if lights on all lanes are changed
            env.traci_connection.trafficlight.setRedYellowGreenState(
                tlsID=edge_id, state=state)
        else:
            # if lights on a single lane is changed
            env.traci_connection.trafficlight.setLinkState(
                tlsID=edge_id, tlsLinkIndex=lane_index, state=state)

    def get_state(self, edge_id):
        """
        Returns the state of the traffic light(s) at the specified edge

        Parameters
        ----------
        edge_id: str
            name of the edge

        Returns
        -------
        state : list <str>
            Index = lane index
            Element = state of the traffic light at that edge/lane
        """
        return self.__tls[edge_id][tc.VAR_EDGES]  # FIXME
