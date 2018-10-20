"""Contains the scenario class for .net.xml files."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario

from lxml import etree
import xml.etree.ElementTree as ElementTree


class NetFileScenario(Scenario):
    """Class that creates a scenario from a .net.xml file."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a scenario from a .net.xml file.

        See flow/scenarios/base_scenario.py for description of params.
        """
        vehicle_data, type_data = self.vehicle_infos("/Users/lucasfischer/sumo/LuSTScenario/scenario/DUARoutes/local.1.rou.xml")

        for t in type_data:
            vehicles.add(t, num_vehicles=type_data[t])

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def vehicle_infos(self,filename):
        """Import of vehicle from a configuration file.

        This is a utility function for computing vehicle information. It imports a
        network configuration file, and returns the information on the vehicle and add it into the Vehicle object

        Parameters
        ----------
        filename : str type
        path to the xml file to load

        Returns
        -------
        Flow Vehicle object
        vehicle_data : dict <dict>
        Key = id of the vehicle
        Element = dict of departure speed, vehicle type, depart Position, depart edges

        """
        # import the .net.xml file containing all edge/type data
        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)

        root = tree.getroot()

        vehicle_data = dict()
        type_data = dict()

        for vehicle in root.findall('vehicle'):

            id_vehicle=vehicle.attrib['id']
            departSpeed=vehicle.attrib['departSpeed']
            depart=vehicle.attrib['depart']
            type_vehicle=vehicle.attrib['type']
            departPos=vehicle.attrib['departPos']
            depart_edges=vehicle.findall('route')[0].attrib["edges"].split(' ')[0]

            if type_vehicle not in type_data:
                type_data[type_vehicle] = 1
            else:
                type_data[type_vehicle] += 1

            vehicle_data[id_vehicle]={'departSpeed':departSpeed,'depart':depart,'type_vehicle':type_vehicle,'departPos':departPos,'depart_edges':depart_edges}

        return vehicle_data, type_data

    def specify_edge_starts(self):
        """See parent class.

        The edge starts are specified from the network configuration file. Note
        that, the values are arbitrary but do not allow the positions of any
        two edges to overlap, thereby making them compatible with all starting
        position methods for vehicles.
        """
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in self._edge_list:
            # the current edge starts (in 1D position) where the last edge
            # ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self._edges[edge_id]["length"]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class.

        All internal edge starts are given a position of -1. This may be
        overridden; however, in general we do not worry about internal edges
        and junctions in large networks.
        """
        return [(":", -1)]

    def close(self):
        """See parent class.

        The close method is overwritten here because we do not want Flow to
        delete externally designed networks.
        """
        pass
