import traci.constants as tc


class TrafficLights:

    def __init__(self):
        """
        Base traffic light class used to place traffic lights in the network
        and describe the state of these traffic lights. In addition, this class
        supports modifying the states of certain lights via TraCI.
        """
        self.__tls = dict()  # contains current time step traffic light data
        self.__ids = list()  # names of nodes with traffic lights
        self.__tls_properties = dict()  # traffic light xml properties
        self.num_traffic_lights = 0  # number of traffic light nodes

    def add(self,
            node_id,
            tls_type="static",
            programID=None,
            offset=None,
            phases=None):
        """
        Adds a traffic light component to the network. When generating networks
        using xml files, this will explicitly place the traffic light in the
        requested node.

        Parameters
        ----------
        node_id : str
            name of the node with traffic lights
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
        # add the node id to the list of controlled nodes
        self.__ids.append(node_id)

        # prepare the data needed to generate xml files
        self.__tls_properties[node_id] = {"id": node_id, "type": tls_type}

        if programID is not None:
            self.__tls_properties[node_id]["programID"] = programID

        if offset is not None:
            self.__tls_properties[node_id]["offset"] = offset

        if phases is not None:
            self.__tls_properties[node_id]["phase"] = phases

    def update(self, tls_subscriptions):
        """
        Updates the states and phases of the traffic lights to match current
        traffic light data.

        Parameters
        ----------
        tls_subscriptions : dict
            sumo traffic light subscription data
        """
        self.__tls = tls_subscriptions.copy()

    def get_ids(self):
        """
        Returns the names of all nodes with traffic lights.
        """
        return self.__ids

    def get_properties(self):
        """
        Returns traffic light properties. Meant to be used by the generator to
        import traffic light data to the .net.xml file
        """
        return self.__tls_properties

    def set_state(self, node_id, state, env, link_index="all"):
        """
        Sets the state of the traffic lights on a specific node.

        Parameters
        ----------
        node_id : str
            name of the node with the controlled traffic lights
        state : str
            requested state(s) for the traffic light
        env : flow.envs.base_env.Env type
            the environment at the current time step
        link_index : int, optional
            index of the link whose traffic light state is meant to be changed.
            If no value is provided, the lights on all links are updated.
        """
        if link_index == "all":
            # if lights on all lanes are changed
            env.traci_connection.trafficlights.setRedYellowGreenState(
                tlsID=node_id, state=state)
        else:
            # if lights on a single lane is changed
            env.traci_connection.trafficlights.setLinkState(
                tlsID=node_id, tlsLinkIndex=link_index, state=state)

    def get_state(self, node_id):
        """
        Returns the state of the traffic light(s) at the specified node

        Parameters
        ----------
        node_id: str
            name of the node

        Returns
        -------
        state : str
            Index = lane index
            Element = state of the traffic light at that node/lane
        """
        return self.__tls[node_id][tc.TL_RED_YELLOW_GREEN_STATE]
