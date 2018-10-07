"""Contains the base scenario class."""

import logging
import random
import numpy as np
import time
import os

try:
    # Import serializable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError:
    Serializable = object

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

VEHICLE_LENGTH = 5  # length of vehicles in the network, in meters


class Scenario(Serializable):
    """Base scenario class.

    Initializes a new scenario. Scenarios are used to specify features of
    a network, including the positions of nodes, properties of the edges
    and junctions connecting these nodes, properties of vehicles and
    traffic lights, and other features as well.

    Several network specific features can be acquired from this class via a
    plethora of get methods (see documentation).

    This class can be instantiated once and reused in multiple experiments.
    Note that this function stores all the relevant parameters. The
    generate() function still needs to be called separately.
    """

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Instantiate the base scenario class.

        Attributes
        ----------
        name : str
            A tag associated with the scenario
        generator_class : Generator type
            Class for generating configuration and net files with placed
            vehicles, e.g. CircleGenerator
        vehicles : Vehicles type
            see flow/core/vehicles.py
        net_params : NetParams type
            see flow/core/params.py
        initial_config : InitialConfig type
            see flow/core/params.py
        traffic_lights : flow.core.traffic_lights.TrafficLights type
            see flow/core/traffic_lights.py
        """
        # Invoke serializable if using rllab
        if Serializable is not object:
            Serializable.quick_init(self, locals())

        self.orig_name = name  # To avoid repeated concatenation upon reset
        self.name = name + time.strftime("_%Y%m%d-%H%M%S") + str(time.time())

        self.generator_class = generator_class
        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config
        self.traffic_lights = traffic_lights

        # create a generator instance
        self.generator = self.generator_class(self.net_params, self.name)

        # create the network configuration file from the generator
        self._edges, self._connections = self.generator.generate_net(
            self.net_params, self.traffic_lights)

        # list of edges and internal links (junctions)
        self._edge_list = [
            edge_id for edge_id in self._edges.keys() if edge_id[0] != ":"
        ]
        self._junction_list = list(
            set(self._edges.keys()) - set(self._edge_list))

        # maximum achievable speed on any edge in the network
        self.max_speed = max(
            self.speed_limit(edge) for edge in self.get_edge_list())

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

        # length of the network, or the portion of the network in
        # which cars are meant to be distributed
        # (may be overridden by subclass __init__())
        if not hasattr(self, "length"):
            self.length = sum([
                self.edge_length(edge_id) for edge_id in self.get_edge_list()
            ])

        # generate starting position for vehicles in the network
        kwargs = initial_config.additional_params
        positions, lanes, speeds = self.generate_starting_positions(
            num_vehicles=vehicles.num_vehicles,
            **kwargs
        )

        # create the sumo configuration files using the generator class
        cfg_name = self.generator.generate_cfg(self.net_params,
                                               self.traffic_lights)

        shuffle = initial_config.shuffle
        self.generator.make_routes(self, positions, lanes, speeds, shuffle)

        # specify the location of the sumo configuration file
        self.cfg = self.generator.cfg_path + cfg_name

    def specify_edge_starts(self):
        """Define edge starts for road sections in the network.

        This is meant to provide some global reference frame for the road
        edges in the network.

        MUST BE implemented in any new scenario subclass.

        Returns
        -------
        edgestarts : list
            list of edge names and starting positions,
            ex: [(edge0, pos0), (edge1, pos1), ...]
        """
        raise NotImplementedError

    def specify_intersection_edge_starts(self):
        """Define edge starts for intersections.

        This is meant to provide some global reference frame for the
        intersections in the network.

        This does not need to be specified if no intersections exist. These
        values can be used to determine the distance of some agent from the
        nearest and/or all intersections.

        Returns
        -------
        intersection_edgestarts : list
            list of intersection names and starting positions,
            ex: [(intersection0, pos0), (intersection1, pos1), ...]
        """
        return []

    def specify_internal_edge_starts(self):
        """Define the edge starts for internal edge nodes.

        This is meant to provide some global reference frame for the internal
        edges in the network.

        These edges are the result of finite-length connections between road
        sections. This methods does not need to be specified if "no-internal-
        links" is set to True in net_params.

        Returns
        -------
        internal_edgestarts : list
            list of internal junction names and starting positions,
            ex: [(internal0, pos0), (internal1, pos1), ...]
        """
        return []

    def get_edge(self, x):
        """Compute an edge and relative position from an absolute position.

        Parameters
        ----------
        x : float
            absolute position in network

        Returns
        -------
        edge position : tup
            1st element: edge name (such as bottom, right, etc.)
            2nd element: relative position on edge
        """
        for (edge, start_pos) in reversed(self.total_edgestarts):
            if x >= start_pos:
                return edge, x - start_pos

    def get_x(self, edge, position):
        """Return the absolute position on the track.

        Parameters
        ----------
        edge : str
            name of the edge
        position : float
            relative position on the edge

        Returns
        -------
        absolute_position : float
            position with respect to some global reference
        """
        # if there was a collision which caused the vehicle to disappear,
        # return an x value of -1001
        if len(edge) == 0:
            return -1001

        if edge[0] == ":":
            try:
                return self.internal_edgestarts_dict[edge] + position
            except KeyError:
                # in case several internal links are being generalized for
                # by a single element (for backwards compatibility)
                edge_name = edge.rsplit("_", 1)[0]
                return self.total_edgestarts_dict.get(edge_name, -1001)
        else:
            return self.total_edgestarts_dict[edge] + position

    def generate_starting_positions(self, num_vehicles=None, **kwargs):
        """Generate starting positions for vehicles in the network.

        Calls all other starting position generating classes.

        Parameters
        ----------
        num_vehicles : int, optional
            number of vehicles to be placed on the network. If no value is
            specified, the value is collected from the vehicles class
        kwargs : dict
            additional arguments that may be updated beyond initial
            configurations, such as modifying the starting position

        Returns
        -------
        startpositions : list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
            list of start lanes
        startvel : list of float
            list of start speeds
        """
        num_vehicles = num_vehicles or self.vehicles.num_vehicles

        if self.initial_config.spacing == "uniform":
            startpositions, startlanes, startvel = self.gen_even_start_pos(
                self.initial_config, num_vehicles, **kwargs)
        elif self.initial_config.spacing == "random":
            startpositions, startlanes, startvel = self.gen_random_start_pos(
                self.initial_config, num_vehicles, **kwargs)
        elif self.initial_config.spacing == "custom":
            startpositions, startlanes, startvel = self.gen_custom_start_pos(
                self.initial_config, num_vehicles, **kwargs)
        else:
            raise ValueError('"spacing" argument in initial_config does not '
                             'contain a valid option')

        return startpositions, startlanes, startvel

    def gen_even_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate uniformly spaced starting positions.

        If the perturbation term in initial_config is set to some positive
        value, then the start positions are perturbed from a uniformly spaced
        distribution by a gaussian whose std is equal to this perturbation
        term.

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
        startvel : list of float
            list of start speeds
        """
        (x0, min_gap, bunching, lanes_distr, available_length,
         available_edges, initial_config) = \
            self._get_start_pos_util(initial_config, num_vehicles, **kwargs)

        increment = available_length / num_vehicles

        # if not all lanes are equal, then we must ensure that vehicles are in
        # two edges at the same time
        flag = False
        lanes = [self.num_lanes(edge) for edge in self.get_edge_list()]
        if any(lanes[0] != lanes[i] for i in range(1, len(lanes))):
            flag = True

        x = x0
        car_count = 0
        startpositions, startlanes = [], []

        # generate uniform starting positions
        while car_count < num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x)

            # ensures that vehicles are not placed in an internal junction
            while pos[0] in dict(self.internal_edgestarts).keys():
                # find the location of the internal edge in total_edgestarts,
                # which has the edges ordered by position
                edges = [tup[0] for tup in self.total_edgestarts]
                indx_edge = next(
                    i for i, edge in enumerate(edges) if edge == pos[0])

                # take the next edge in the list, and place the car at the
                # beginning of this edge
                if indx_edge == len(edges) - 1:
                    next_edge_pos = self.total_edgestarts[0]
                else:
                    next_edge_pos = self.total_edgestarts[indx_edge + 1]

                x = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            # ensures that you are in an acceptable edge
            while pos[0] not in available_edges:
                x = (x + self.edge_length(pos[0])) % self.length
                pos = self.get_edge(x)

            # ensure that in variable lane settings vehicles always start a
            # vehicle's length away from the start of the edge. This, however,
            # prevents the spacing to be completely uniform.
            if flag and pos[1] < VEHICLE_LENGTH:
                pos0, pos1 = pos
                pos = (pos0, VEHICLE_LENGTH)
                x += VEHICLE_LENGTH
                increment -= (VEHICLE_LENGTH * self.num_lanes(pos0)) / \
                             (num_vehicles - car_count)

            # place vehicles side-by-side in all available lanes on this edge
            for lane in range(min([self.num_lanes(pos[0]), lanes_distr])):
                car_count += 1
                startpositions.append(pos)
                startlanes.append(lane)

                if car_count == num_vehicles:
                    break

            x = (x + increment + VEHICLE_LENGTH + min_gap) % self.length

        # add a perturbation to each vehicle, while not letting the vehicle
        # leave its current edge
        if initial_config.perturbation > 0:
            for i in range(num_vehicles):
                perturb = np.random.normal(0, initial_config.perturbation)
                edge, pos = startpositions[i]
                pos = max(0, min(self.edge_length(edge), pos + perturb))
                startpositions[i] = (edge, pos)

        # all vehicles start with an initial speed of 0 m/s
        startvel = [0 for _ in range(len(startlanes))]

        return startpositions, startlanes, startvel

    def gen_random_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate random starting positions.

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
        startvel : list of float
            list of start speeds
        """
        (x0, min_gap, bunching, lanes_distr, available_length,
         available_edges, initial_config) = self._get_start_pos_util(
            initial_config, num_vehicles, **kwargs)

        # extra space a vehicle needs to cover from the start of an edge to be
        # fully in the edge and not risk having a gap with a vehicle behind it
        # that is smaller than min_gap
        efs = min_gap + VEHICLE_LENGTH  # extra front space

        for edge in available_edges:
            available_length -= efs * min([self.num_lanes(edge), lanes_distr])

        # choose random positions for each vehicle
        init_absolute_pos = \
            [random.random() * available_length
             for _ in range(num_vehicles)]

        # sort the positions of vehicles, for simplicity in using
        init_absolute_pos.sort()

        # these positions do not include the length of the vehicle, which need
        # to be added
        for i in range(num_vehicles):
            init_absolute_pos[i] += (VEHICLE_LENGTH + min_gap) * i

        decrement = 0
        edge_indx = 0
        startpositions = []
        startlanes = []
        for i in range(num_vehicles):
            edge_i = available_edges[edge_indx]
            pos_i = (init_absolute_pos[i] - decrement) % (
                    self.edge_length(edge_i) - efs)
            lane_i = int(((init_absolute_pos[i] - decrement) - pos_i) /
                         (self.edge_length(edge_i) - efs))

            pos_i += efs

            while lane_i > min([self.num_lanes(edge_i), lanes_distr]) - 1:
                decrement += min([self.num_lanes(edge_i), lanes_distr]) \
                             * (self.edge_length(edge_i) - efs)
                edge_indx += 1

                edge_i = available_edges[edge_indx]
                pos_i = (init_absolute_pos[i] - decrement) % (
                        self.edge_length(edge_i) - efs)

                lane_i = int(((init_absolute_pos[i] - decrement) - pos_i) /
                             (self.edge_length(edge_i) - efs))

                pos_i += efs

            startpositions.append((edge_i, pos_i))
            startlanes.append(lane_i)

        # all vehicles start with an initial speed of 0 m/s
        startvel = [0 for _ in range(len(startlanes))]

        return startpositions, startlanes, startvel

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate a user defined set of starting positions.

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
        startvel : list of float
            list of start speeds
        """
        raise NotImplementedError

    def _get_start_pos_util(self, initial_config, num_vehicles, **kwargs):
        """Prepare initial_config data for starting position methods.

        Performs some pre-processing to the initial_config and **kwargs terms,
        and returns the necessary values for all starting position generating
        functions.

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
        x0 : float
            starting position of the first vehicle, in meters
        min_gap : float
            minimum gap between vehicles
        bunching : float
            the amount of space freed up in the network (per lane)
        lanes_distribution : int
            number of lanes the vehicles are supposed to be distributed over
        available_length : float
            total available free space for vehicle to be placed, over all lanes
            within the distributable lanes, in meters
        initial_config : InitialConfig type
            modified version of the initial_config parameter

        Raises
        ------
        ValueError
            If there is not enough space to place all vehicles in the allocated
            space in the network with the specified minimum gap.
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts, and so
        # overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = initial_config.bunching
        # check if requested bunching value is not valid (negative)
        if bunching < 0:
            logging.warning('"bunching" cannot be negative; setting to 0')
            bunching = 0
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        # compute the lanes distribution (adjust of edge cases)
        if initial_config.edges_distribution == "all":
            max_lane = max(
                [self.num_lanes(edge_id) for edge_id in self.get_edge_list()])
        else:
            max_lane = max([
                self.num_lanes(edge_id)
                for edge_id in initial_config.edges_distribution
            ])

        if initial_config.lanes_distribution > max_lane:
            lanes_distribution = max_lane
        elif initial_config.lanes_distribution < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        else:
            lanes_distribution = initial_config.lanes_distribution

        if initial_config.edges_distribution == "all":
            distribution_length = \
                sum([self.edge_length(edge_id) *
                     min([self.num_lanes(edge_id), lanes_distribution])
                     for edge_id in self.get_edge_list()])
        else:
            distribution_length = \
                sum([self.edge_length(edge_id) *
                     min([self.num_lanes(edge_id), lanes_distribution])
                     for edge_id in initial_config.edges_distribution])

        min_gap = max(0, initial_config.min_gap)

        if initial_config.edges_distribution == "all":
            available_edges = self.get_edge_list()
        else:
            available_edges = initial_config.edges_distribution

        available_length = \
            distribution_length - lanes_distribution * bunching - \
            num_vehicles * (min_gap + VEHICLE_LENGTH)

        if available_length < 0:
            raise ValueError("There is not enough space to place all vehicles "
                             "in the network.")

        return (x0, min_gap, bunching, lanes_distribution, available_length,
                available_edges, initial_config)

    def edge_length(self, edge_id):
        """Return the length of a given edge/junction.

        Return -1001 if edge not found.
        """
        try:
            return self._edges[edge_id]["length"]
        except KeyError:
            print('Error in edge length with key', edge_id)
            return -1001

    def speed_limit(self, edge_id):
        """Return the speed limit of a given edge/junction.

        Return -1001 if edge not found.
        """
        try:
            return self._edges[edge_id]["speed"]
        except KeyError:
            print('Error in speed limit with key', edge_id)
            return -1001

    def num_lanes(self, edge_id):
        """Return the number of lanes of a given edge/junction.

        Return -1001 if edge not found.
        """
        try:
            return self._edges[edge_id]["lanes"]
        except KeyError:
            print('Error in num lanes with key', edge_id)
            return -1001

    def get_edge_list(self):
        """Return the names of all edges in the network."""
        return self._edge_list

    def get_junction_list(self):
        """Return the names of all junctions in the network."""
        return self._junction_list

    def next_edge(self, edge, lane):
        """Return the next edge/lane pair from the given edge/lane.

        These edges may also be internal links (junctions). Returns an empty
        list if there are no edge/lane pairs in front.
        """
        try:
            return self._connections["next"][edge][lane]
        except KeyError:
            return []

    def prev_edge(self, edge, lane):
        """Return the edge/lane pair right before this edge/lane.

        These edges may also be internal links (junctions). Returns an empty
        list if there are no edge/lane pairs behind.
        """
        try:
            return self._connections["prev"][edge][lane]
        except KeyError:
            return []

    def close(self):
        """Close the scenario class.

        Deletes the xml files that were created by the generator class. This
        is to prevent them from building up in the debug folder.
        """
        os.remove(self.generator.net_path + self.generator.nodfn)
        os.remove(self.generator.net_path + self.generator.edgfn)
        os.remove(self.generator.net_path + self.generator.cfgfn)
        os.remove(self.generator.cfg_path + self.generator.addfn)
        os.remove(self.generator.cfg_path + self.generator.guifn)
        os.remove(self.generator.cfg_path + self.generator.netfn)
        os.remove(self.generator.cfg_path + self.generator.roufn)
        os.remove(self.generator.cfg_path + self.generator.sumfn)

        # the connection file is not always created
        try:
            os.remove(self.generator.net_path + self.generator.confn)
        except OSError:
            pass

        # neither is the type file
        try:
            os.remove(self.generator.net_path + self.generator.typfn)
        except OSError:
            pass

    def __str__(self):
        """Return the name of the scenario and the number of vehicles."""
        return "Scenario " + self.name + " with " + \
               str(self.vehicles.num_vehicles) + " vehicles."
