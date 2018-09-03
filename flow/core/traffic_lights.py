"""Contains the traffic light class."""

import traci.constants as tc

# DEFAULTS
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.8
SHOW_DETECTORS = True


class TrafficLights:
    """Base traffic light.

    This class is used to place traffic lights in the network and describe
    the state of these traffic lights. In addition, this class supports
    modifying the states of certain lights via TraCI.
    """

    def __init__(self, baseline=False):
        """Instantiate base traffic light.

        Parameters
        ----------
        baseline: bool
        """
        self.__tls = dict()  # contains current time step traffic light data
        self.__ids = list()  # names of nodes with traffic lights
        self.__tls_properties = dict()  # traffic light xml properties
        self.num_traffic_lights = 0  # number of traffic light nodes

        # all traffic light parameters are set to default baseline values
        self.baseline = baseline

    def add(self,
            node_id,
            tls_type="static",
            programID=10,
            offset=None,
            phases=None,
            maxGap=None,
            detectorGap=None,
            showDetectors=None,
            file=None,
            freq=None):
        """Add a traffic light component to the network.

        When generating networks using xml files, using this method to add a
        traffic light will explicitly place the traffic light in the requested
        node of the generated network.

        If traffic lights are not added here but are already present in the
        network (e.g. through a prebuilt net.xml file), then the traffic light
        class will identify and add them separately.

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

            * "duration": length of the current phase cycle (in sec)
            * "state": string consist the sequence of states in the phase
            * "minDur": optional
                The minimum duration of the phase when using type actuated
            * "maxDur": optional
                The maximum duration of the phase when using type actuated

        maxGap : int, used for actuated traffic lights
            describes the maximum time gap between successive vehicle that
            will cause the current phase to be prolonged
        detectorGap : int, used for actuated traffic lights
            determines the time distance between the (automatically generated)
            detector and the stop line in seconds (at each lanes maximum speed)
        showDetectors : bool, used for actuated traffic lights
            toggles whether or not detectors are shown in sumo-gui
        file : str, optional
            which file the detector shall write results into
        freq : int, optional
            the period over which collected values shall be aggregated

        Note
        ----
        For information on defining traffic light properties, see:
        http://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
        """
        # increment the number of traffic lights
        self.num_traffic_lights += 1

        # TODO add proper checks here: make sure programID exists
        # NOTE: the keys you add to the dictionary need to match the xml spec
        # add the node id to the list of controlled nodes
        self.__ids.append(node_id)

        # prepare the data needed to generate xml files
        self.__tls_properties[node_id] = {"id": node_id, "type": tls_type}

        if programID:
            self.__tls_properties[node_id]["programID"] = programID

        if offset:
            self.__tls_properties[node_id]["offset"] = offset

        if phases:
            self.__tls_properties[node_id]["phases"] = phases

        if tls_type == "actuated":
            # Required parameters
            self.__tls_properties[node_id]["max-gap"] = \
                maxGap if maxGap else MAX_GAP
            self.__tls_properties[node_id]["detector-gap"] = \
                detectorGap if detectorGap else DETECTOR_GAP
            self.__tls_properties[node_id]["show-detectors"] = \
                showDetectors if showDetectors else SHOW_DETECTORS

            # Optional parameters
            if file:
                self.__tls_properties[node_id]["file"] = file

            if freq:
                self.__tls_properties[node_id]["freq"] = freq

    def update(self, tls_subscriptions):
        """Update the states and phases of the traffic lights.

        This is called by the environment class, and ensures that the traffic
        light variables match current traffic light data.

        Parameters
        ----------
        tls_subscriptions : dict
            sumo traffic light subscription data
        """
        self.__tls = tls_subscriptions.copy()

    def get_ids(self):
        """Return the names of all nodes with traffic lights."""
        return self.__ids

    def get_properties(self):
        """Return traffic light properties.

        This is meant to be used by the generator to import traffic light data
        to the .net.xml file
        """
        return self.__tls_properties

    def set_state(self, node_id, state, env, link_index="all"):
        """Set the state of the traffic lights on a specific node.

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
            env.traci_connection.trafficlight.setRedYellowGreenState(
                tlsID=node_id, state=state)
        else:
            # if lights on a single lane is changed
            env.traci_connection.trafficlight.setLinkState(
                tlsID=node_id, tlsLinkIndex=link_index, state=state)

    def get_state(self, node_id):
        """Return the state of the traffic light(s) at the specified node.

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

    def actuated_default(self):
        """
        Return the default values to be used for the generator
        for a system where all junctions are actuated traffic lights.

        Returns
        -------
        tl_logic: dict
        """
        tl_type = "actuated"
        program_id = 1
        max_gap = 3.0
        detector_gap = 0.8
        show_detectors = True
        phases = [{
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "GGGrrrGGGrrr"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "yyyrrryyyrrr"
        }, {
            "duration": "31",
            "minDur": "8",
            "maxDur": "45",
            "state": "rrrGGGrrrGGG"
        }, {
            "duration": "6",
            "minDur": "3",
            "maxDur": "6",
            "state": "rrryyyrrryyy"
        }]

        return {
            "tl_type": str(tl_type),
            "program_id": str(program_id),
            "max_gap": str(max_gap),
            "detector_gap": str(detector_gap),
            "show_detectors": show_detectors,
            "phases": phases
        }
