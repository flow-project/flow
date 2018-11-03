"""Contains the generator class for .net.xml files."""

from flow.core.generator import Generator
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles
from lxml import etree
import xml.etree.ElementTree as ElementTree

# Added by Crystal for extracting routes
import re


class NetFileGenerator(Generator):
    """Class used to generate network files from a .net.xml file.

    The .net.xml file is specified in the NetParams object. For example:

    >>> from flow.core.params import NetParams
    >>> net_params = NetParams(netfile="/path/to/netfile.net.xml")

    No "specify_nodes" and "specify_edges" routes are needed. However, a
    "specify_routes" file is still needed to specify the appropriate routes
    vehicles can traverse in the network.
    """

    def generate_net(self, net_params, traffic_lights):
        """See parent class.

        The network file is generated from the .osm file specified in
        net_params.osm_path
        """
        # name of the .net.xml file (located in cfg_path)
        self.netfn = "/Users/lucasfischer/sumo/LuSTScenario/scenario/lust.net.xml"
        # self.netfn = net_params.netfile

        # collect data from the generated network configuration file
        edges_dict, conn_dict = self._import_edges_from_net()

        self.edge_dict = edges_dict
        return edges_dict, conn_dict

    def specify_nodes(self, net_params):
        """See class definition."""
        pass

    def specify_edges(self, net_params):
        """See class definition."""
        pass

    def specify_types(self, net_params):
        types = self.vehicle_type("//Users/lucasfischer/sumo/LuSTScenario/scenario/vtypes.add.xml")
        return types

    def _import_routes_from_net(self, filename):
        """Import route from a configuration file.

        This is a utility function for computing route information. It imports a
        network configuration file, and returns the information on the routes
        taken by all the vehicle located in the file.

        Parameters
        ----------
        filename : str type
        path to the xml file to load

        Returns
        -------
        routes_data : dict <dict>
        Key = name of the first route taken
        Element = list of all the routes taken by the vehicle starting in that route

        """
        # # import the .net.xml file containing all edge/type data
        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)

        root = tree.getroot()

        # Collect information on the available types (if any are available).
        # This may be used when specifying some route data.
        routes_data = dict()

        for vehicle in root.findall('vehicle'):
            for route in vehicle.findall('route'):
                route_edges = route.attrib["edges"].split(' ')
                key=route_edges[0]
                if key in routes_data.keys():
                    for routes in routes_data[key]:
                        if routes==route_edges:
                            pass
                        else:
                            routes_data[key].append(route_edges)
                else:
                    routes_data[key] = [route_edges]
        return routes_data


        # Crystal's version of extracting routes
        # I didn't test this myself and I don't guarantee correctness in this
        # (completeness or uniqueness of routes), but here it is.
        # Use for faster performance and be sure to test it.

        with open(filename, 'r') as myfile:
            data = myfile.read()

            # regex to extract routes
            pattern = '<route edges=".*?"\/>' # Extracts <route edges="..."/>
            matches = re.findall(pattern, data)

            # Strip <route edges>
            for i in range(len(matches)):
                matches[i] = matches[i].split('"')[1]

            # Create each as list and add to set
            distinct_routes = set()
            for i in range(len(matches)):
                curr_route = tuple(matches[i].split(' '))
                matches[i] = curr_route
                distinct_routes.add(curr_route)

            # Compare length of all routes (matches) vs length of distinct routes
            #print(len(matches), len(distinct_routes))

            routes_data = {}
            for route in distinct_routes:
                first_edge = route[0]
                if first_edge in routes_data:

                    ### We need to figure out that part
                    pass
                    #routes_data[first_edge].append(list(route))
                else:

                    routes_data[first_edge] = list(route)
                    #routes_data[first_edge] = [list(route)]

            # Print first item in edge-routes dictionary
            #print(list(routes_data.items())[0])

            return routes_data



    def specify_routes(self, net_params):
        """ Format all the routes from the xml file
        Parameters
        ----------
        filename : str type
        path to the rou.xml file to load
        Returns
        -------
        routes_data : dict <dict>
        Key = name of the first route taken
        Element = list of all the routes taken by the vehicle starting in that route
        """
        routes_data={}
        for edge in self.edge_dict:
            if ':' not in edge:
                routes_data[edge]=[edge]
        #routes_data = self._import_routes_from_net("/Users/lucasfischer/sumo/LuSTScenario/scenario/DUARoutes/local.1.rou.xml")
        return routes_data

    def _import_tls_from_net(self,filename):
        """Import traffic lights from a configuration file.
        This is a utility function for computing traffic light information. It imports a
        network configuration file, and returns the information of the traffic lights in the file.
        Parameters
        ----------
        filename : str type
        path to the rou.xml file to load
        Returns
        -------
        tl_logic : TrafficLights
        """
        # import the .net.xml file containing all edge/type data
        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)
        root = tree.getroot()
        # create TrafficLights() class object to store traffic lights information from the file
        tl_logic = TrafficLights()
        for tl in root.findall('tlLogic'):
            phases = [phase.attrib for phase in tl.findall('phase')]
            tl_logic.add(tl.attrib['id'], tl.attrib['type'], tl.attrib['programID'], tl.attrib['offset'], phases)
        return tl_logic



    def vehicle_type(self,filename):
        """Import vehicle type from an vtypes.add.xml file .
        This is a utility function for outputting all the type of vehicle . .
        Parameters
        ----------
        filename : str type
        path to the vtypes.add.xml file to load
        Returns
        -------
        dict: the key is the vehicle_type id and the value is a dict we've type of the vehicle, depart edges , depart Speed, departPos
        """
        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)

        root = tree.getroot()
        veh_type = {}

        for transport in root.findall('vTypeDistribution'):
            for vtype in transport.findall('vType'):
                vClass=vtype.attrib['vClass']
                id_type_vehicle=vtype.attrib['id']
                accel=vtype.attrib['accel']
                decel=vtype.attrib['decel']
                sigma=vtype.attrib['sigma']
                length=vtype.attrib['length']
                minGap=vtype.attrib['minGap']
                maxSpeed=vtype.attrib['maxSpeed']
                probability=vtype.attrib['probability']
                speedDev=vtype.attrib['speedDev']
                veh_type[id_type_vehicle]={'vClass':vClass,'accel':accel,'decel':decel,'sigma':sigma,'length':length,'minGap':minGap,'maxSpeed':maxSpeed,'probability':probability,'speedDev':speedDev}
        return veh_type


    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate a user defined set of starting positions.
        This method is just used for testing.
        Parameters
        ----------
        initial_config : InitialConfig type
        see flow/core/params.py
        num_vehicles : int
        number of vehicles to be placed on the network
        kwargs : dict
        extra components, usually defined during reset to overwrite initial
        config parameters
        Returns
        -------
        startpositions : list of tuple (float, float)
        list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
        list of start lanes
        """
        return kwargs["start_positions"], kwargs["start_lanes"]
