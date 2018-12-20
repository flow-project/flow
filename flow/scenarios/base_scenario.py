"""Contains the base scenario class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import time

try:
    # Import serializable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError:
    Serializable = object


class Scenario(Serializable):
    """Base scenario class.

    Initializes a new scenario. Scenarios are used to specify features of
    a network, including the positions of nodes, properties of the edges
    and junctions connecting these nodes, properties of vehicles and
    traffic lights, and other features as well.
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the base scenario class.

        Attributes
        ----------
        name : str
            A tag associated with the scenario
        vehicles : Vehicles type
            see flow/core/kernel/vehicle/base.py
        net_params : NetParams type
            see flow/core/params.py
        initial_config : InitialConfig type
            see flow/core/params.py
        traffic_lights : flow.core.params.TrafficLightParams
            see flow/core/params.py
        """
        # Invoke serializable if using rllab
        if Serializable is not object:
            Serializable.quick_init(self, locals())

        self.orig_name = name  # To avoid repeated concatenation upon reset
        self.name = name + time.strftime("_%Y%m%d-%H%M%S") + str(time.time())

        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config
        self.traffic_lights = traffic_lights

        # specify the attributes of the nodes
        self.nodes = self.specify_nodes(net_params)

        # collect the attributes of each edge
        self.edges = self.specify_edges(net_params)

        # specify the types attributes (default is None)
        self.types = self.specify_types(net_params)

        # specify the connection attributes (default is None)
        self.connections = self.specify_connections(net_params)

        # specify routes vehicles can take
        self.routes = self.specify_routes(net_params)

        # parameters to be specified under each unique subclass's
        # __init__() function
        self.edgestarts = self.specify_edge_starts()

        # these optional parameters need only be used if "no-internal-links"
        # is set to "false" while calling sumo's netconvert function
        self.internal_edgestarts = self.specify_internal_edge_starts()
        self.intersection_edgestarts = self.specify_intersection_edge_starts()

        # in case the user did not write the intersection edge-starts in
        # internal edge-starts as well (because of redundancy), merge the two
        # together
        self.internal_edgestarts += self.intersection_edgestarts
        seen = set()
        self.internal_edgestarts = \
            [item for item in self.internal_edgestarts
             if item[1] not in seen and not seen.add(item[1])]
        self.internal_edgestarts_dict = dict(self.internal_edgestarts)

        # total_edgestarts and total_edgestarts_dict contain all of the above
        # edges, with the former being ordered by position
        if self.net_params.no_internal_links:
            self.total_edgestarts = self.edgestarts
        else:
            self.total_edgestarts = self.edgestarts + self.internal_edgestarts
        self.total_edgestarts.sort(key=lambda tup: tup[1])

        self.total_edgestarts_dict = dict(self.total_edgestarts)

    # TODO: convert to property
    def specify_edge_starts(self):
        """Define edge starts for road sections in the network.

        This is meant to provide some global reference frame for the road
        edges in the network.

        MUST BE implemented in any new scenario subclass.

        Returns
        -------
        list
            list of edge names and starting positions,
            ex: [(edge0, pos0), (edge1, pos1), ...]
        """
        raise NotImplementedError

    # TODO: convert to property
    def specify_intersection_edge_starts(self):
        """Define edge starts for intersections.

        This is meant to provide some global reference frame for the
        intersections in the network.

        This does not need to be specified if no intersections exist. These
        values can be used to determine the distance of some agent from the
        nearest and/or all intersections.

        Returns
        -------
        list
            list of intersection names and starting positions,
            ex: [(intersection0, pos0), (intersection1, pos1), ...]
        """
        return []

    # TODO: convert to property
    def specify_internal_edge_starts(self):
        """Define the edge starts for internal edge nodes.

        This is meant to provide some global reference frame for the internal
        edges in the network.

        These edges are the result of finite-length connections between road
        sections. This methods does not need to be specified if "no-internal-
        links" is set to True in net_params.

        Returns
        -------
        list
            list of internal junction names and starting positions,
            ex: [(internal0, pos0), (internal1, pos1), ...]
        """
        return []

    # TODO: convert to property
    def specify_nodes(self, net_params):
        """Specify the attributes of nodes in the network.

        Parameters
        ----------
        net_params : NetParams type
            see flow/core/params.py

        Returns
        -------
        list of dict

            A list of node attributes (a separate dict for each node). Nodes
            attributes must include:

            * id {string} -- name of the node
            * x {float} -- x coordinate of the node
            * y {float} -- y coordinate of the node

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Node_Descriptions
        """
        raise NotImplementedError

    # TODO: convert to property
    def specify_edges(self, net_params):
        """Specify the attributes of edges connecting pairs on nodes.

        Parameters
        ----------
        net_params : NetParams type
            see flow/core/params.py

        Returns
        -------
        list of dict

            A list of edges attributes (a separate dict for each edge). Edge
            attributes must include:

            * id {string} -- name of the edge
            * from {string} -- name of node the directed edge starts from
            * to {string} -- name of the node the directed edge ends at

            In addition, the attributes must contain at least one of the
            following:

            * "numLanes" {int} and "speed" {float} -- the number of lanes and
              speed limit of the edge, respectively
            * type {string} -- a type identifier for the edge, which can be
              used if several edges are supposed to possess the same number of
              lanes, speed limits, etc...

        Other attributes may also be specified. See:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Edge_Descriptions
        """
        raise NotImplementedError

    # TODO: convert to property
    def specify_types(self, net_params):
        """Specify the attributes of various edge types (if any exist).

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        list of dict
            A list of type attributes for specific groups of edges. If none are
            specified, no .typ.xml file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Type_Descriptions
        """
        return None

    # TODO: convert to property
    def specify_connections(self, net_params):
        """Specify the attributes of connections.

        These attributes are used to describe how any specific node's incoming
        and outgoing edges/lane pairs are connected. If no connections are
        specified, sumo generates default connections.

        Parameters
        ----------
        net_params: NetParams type
            see flow/core/params.py

        Returns
        -------
        list of dict
            A list of connection attributes. If none are specified, no .con.xml
            file is created.

        For information on type attributes, see:
        http://sumo.dlr.de/wiki/Networks/Building_Networks_from_own_XML-descriptions#Connection_Descriptions
        """
        return None

    # TODO: convert to property
    def specify_routes(self, net_params):
        """Specify the routes vehicles can take starting from any edge.

        The routes are specified as lists of edges the vehicle must traverse,
        with the first edge corresponding to the edge the vehicle begins on.
        Note that the edges must be connected for the route to be valid.

        Currently, only one route is allowed from any given starting edge.

        Parameters
        ----------
        net_params : NetParams type
            see flow/core/params.py

        Returns
        -------
        dict
            Key = name of the starting edge
            Element = list of edges a vehicle starting from this edge must
            traverse.
        """
        raise NotImplementedError

    @staticmethod
    def gen_custom_start_pos(cls, initial_config, num_vehicles, **kwargs):
        """Generate a user defined set of starting positions.

        Parameters
        ----------
        cls : flow.core.kernel.scenario.KernelScenario
            flow scenario kernel, with all the relevant methods implemented
        initial_config : flow.core.params.InitialConfig
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network
        kwargs : dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        list of int
            list of start lanes
        list of float
            list of start speeds
        """
        raise NotImplementedError

    def __str__(self):
        """Return the name of the scenario and the number of vehicles."""
        return "Scenario " + self.name + " with " + \
               str(self.vehicles.num_vehicles) + " vehicles."
