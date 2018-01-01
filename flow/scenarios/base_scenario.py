import logging
import random
import numpy as np

try:
    # Import serialiable if rllab is installed
    from rllab.core.serializable import Serializable
except ImportError as e:
    Serializable = object

from flow.core.params import InitialConfig


VEHICLE_LENGTH = 5  # length of vehicles in the network, in meters


class Scenario(Serializable):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig()):
        """
        Abstract base class. Initializes a new scenario. This class can be
        instantiated once and reused in multiple experiments. Note that this
        function stores all the relevant parameters. The generate() function
        still needs to be called separately.

        Attributes
        ----------
        name: str
            A tag associated with the scenario
        generator_class: Generator type
            Class for generating configuration and net files with placed
            vehicles, e.g. CircleGenerator
        vehicles: Vehicles type
            see flow/core/vehicles.py
        net_params: NetParams type
            see flow/core/params.py
        initial_config: InitialConfig type
            see flow/core/params.py
        """
        # Invoke serialiable if using rllab
        if Serializable is not object:
            Serializable.quick_init(self, locals())

        self.name = name
        self.generator_class = generator_class
        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config

        # create a generator instance
        self.generator = self.generator_class(self.net_params, self.name)

        # create the network configuration file from the generator
        self.edges = self.generator.generate_net(self.net_params)

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

        # total_edgestarts and total_edgestarts_dict contain all of the above
        # edges, with the former being ordered by position
        if self.net_params.no_internal_links:
            self.total_edgestarts = self.edgestarts
        else:
            self.total_edgestarts = self.edgestarts + self.internal_edgestarts
        self.total_edgestarts.sort(key=lambda tup: tup[1])

        self.total_edgestarts_dict = dict(self.total_edgestarts)

        # length of the network, or the portion of the network in which cars are
        # meant to be distributed (may be overridden by subclass __init__())
        if not hasattr(self, "length"):
            self.length = sum([self.edge_length(edge_id)
                               for edge_id in self.get_edge_list()])

        # generate starting position for vehicles in the network
        if self.initial_config.positions is None:
            self.initial_config.positions, self.initial_config.lanes = \
                self.generate_starting_positions()

        # create the sumo configuration files using the generator class
        cfg_name = self.generator.generate_cfg(self.net_params)
        self.generator.make_routes(self, self.initial_config)

        # specify the location of the sumo configuration file
        self.cfg = self.generator.cfg_path + cfg_name

    def specify_edge_starts(self):
        """
        Defines edge starts for road sections with respect to some global
        reference frame.
        MUST BE implemented in any new scenario subclass.

        Returns
        -------
        edgestarts: list
            list of edge names and starting positions,
            ex: [(edge0, pos0), (edge1, pos1), ...]
        """
        raise NotImplementedError

    def specify_intersection_edge_starts(self):
        """
        Defines edge starts for intersections with respect to some global
        reference frame. Need not be specified if no intersections exist.
        These values can be used to determine the distance of some agent from
        the nearest and/or all intersections.

        Returns
        -------
        intersection_edgestarts: list
            list of intersection names and starting positions,
            ex: [(intersection0, pos0), (intersection1, pos1), ...]
        """
        return []

    def specify_internal_edge_starts(self):
        """
        Defines the edge starts for internal edge nodes (caused by finite-length
        connections between road sections) with respect to some global reference
        frame. Does not need to be specified if "no-internal-links" is set to
        True in net_params.

        Returns
        -------
        internal_edgestarts: list
            list of internal junction names and starting positions,
            ex: [(internal0, pos0), (internal1, pos1), ...]
        """
        return []

    def generate(self):
        """
        Applies self.generator_class to create a net and corresponding cfg
        files, including placement of vehicles (name.rou.xml).

        Returns
        -------
        cfg: str
            path to configuration (.sumo.cfg) file
        """
        logging.info("Config file not defined, generating using generator")

        self.generator = self.generator_class(self.net_params, self.name)
        self.generator.generate_net(self.net_params)
        cfg_name = self.generator.generate_cfg(self.net_params)
        self.generator.make_routes(self, self.initial_config)

        return self.generator.cfg_path + cfg_name

    def get_edge(self, x):
        """
        Given an absolute position x on the track, returns the edge (name) and
        relative position on that edge.

        Parameters
        ----------
        x: float
            absolute position in network

        Returns
        -------
        edge position: tup
            1st element: edge name (such as bottom, right, etc.)
            2nd element: relative position on edge
        """
        starte = ""
        startx = 0

        for (e, s) in self.total_edgestarts:
            if x >= s:
                starte = e
                startx = x - s

        return starte, startx

    def get_x(self, edge, position):
        """
        Given an edge name and relative position, return the absolute position
        on the track.

        Parameters
        ----------
        edge: str
            name of the edge
        position: float
            relative position on the edge

        Returns
        -------
        absolute_position: float
            position with respect to some global reference
        """
        if edge in dict(self.edgestarts).keys():
            return self.total_edgestarts_dict[edge] + position
        else:
            for edge_tuple in self.internal_edgestarts:
                if edge_tuple[0] in edge:
                    return edge_tuple[1] + position

    def generate_starting_positions(self, **kwargs):
        """
        Generates starting positions for vehicles in the network. Calls all
        other starting position generating classes.

        Parameters
        ----------
        kwargs: dict
            additional arguments that may be updated beyond initial
            configurations, such as modifying the starting position

        Returns
        -------
        startpositions: list
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes: list
            list of start lanes
        """
        if self.initial_config.spacing == "uniform":
            startpositions, startlanes = \
                self.gen_even_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "random":
            startpositions, startlanes = \
                self.gen_random_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "custom":
            startpositions, startlanes = \
                self.gen_custom_start_pos(self.initial_config, **kwargs)
        else:
            raise ValueError('"spacing" argument in initial_config does not '
                             'contain a valid option')

        return startpositions, startlanes

    def gen_even_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are uniformly spaced across the network.
        If the perturbation term in initial_config is set to some positive
        value, then the start positions are perturbed from a uniformly spaced
        distribution by a gaussian whose std is equal to this perturbation term.

        Parameters
        ----------
        initial_config: InitialConfig type
            see flow/core/params.py
        kwargs: dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions: list
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes: list
            list of start lanes
        """
        x0, min_gap, bunching, lanes_distr, available_length, available_edges, \
            initial_config = self._get_start_pos_util(initial_config, **kwargs)

        increment = available_length / self.vehicles.num_vehicles

        # if not all lanes are equal, then we must ensure that vehicles are in
        # two edges at the same time
        flag = False
        lanes = [self.num_lanes(edge) for edge in self.get_edge_list()]
        if any([lanes[0] != lanes[i] for i in range(1, len(lanes))]):
            flag = True

        x = x0
        car_count = 0
        startpositions, startlanes = [], []

        # generate uniform starting positions
        while car_count < self.vehicles.num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x)

            # ensures that vehicles are not placed in an internal junction
            while pos[0] in dict(self.internal_edgestarts).keys():
                # find the location of the internal edge in total_edgestarts,
                # which has the edges ordered by position
                edges = [tup[0] for tup in self.total_edgestarts]
                indx_edge = [i for i in range(len(edges)) if edges[i] == pos[0]][0]

                # take the next edge in the list, and place the car at the
                # beginning of this edge
                if indx_edge == len(edges)-1:
                    next_edge_pos = self.total_edgestarts[0]
                else:
                    next_edge_pos = self.total_edgestarts[indx_edge+1]

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
                             (self.vehicles.num_vehicles - car_count)

            # place vehicles side-by-side in all available lanes on this edge
            for lane in range(min([self.num_lanes(pos[0]), lanes_distr])):
                car_count += 1
                startpositions.append(pos)
                startlanes.append(lane)

                if car_count == self.vehicles.num_vehicles:
                    break

            x = (x + increment + VEHICLE_LENGTH + min_gap) % self.length

        # add a perturbation to each vehicle, while not letting the vehicle
        # leave its current edge
        if initial_config.perturbation > 0:
            for i in range(self.vehicles.num_vehicles):
                perturb = np.random.normal(0, initial_config.perturbation)
                edge, pos = startpositions[i]
                pos = max(0, min(self.edge_length(edge), pos + perturb))
                startpositions[i] = (edge, pos)

        return startpositions, startlanes

    def gen_random_start_pos(self, initial_config, **kwargs):
        """
        Generates random starting positions for vehicles in the allocated lanes
        and edges.

        Parameters
        ----------
        initial_config: InitialConfig type
            see flow/core/params.py
        kwargs: dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions: list
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes: list
            list of start lanes
        """
        x0, min_gap, bunching, lanes_distr, available_length, available_edges, \
        initial_config = self._get_start_pos_util(initial_config, **kwargs)

        # extra space a vehicle needs to cover from the start of an edge to be
        # fully in the edge and not risk having a gap with a vehicle behind it
        # that is smaller than min_gap
        efs = min_gap + VEHICLE_LENGTH  # extra front space

        for edge in available_edges:
            available_length -= efs * min([self.num_lanes(edge), lanes_distr])

        # choose random positions for each vehicle
        init_absolute_pos = \
            [random.random() * available_length
             for _ in range(self.vehicles.num_vehicles)]

        # sort the positions of vehicles, for simplicity in using
        init_absolute_pos.sort()

        # these positions do not include the length of the vehicle, which need
        # to be added
        for i in range(self.vehicles.num_vehicles):
            init_absolute_pos[i] += (VEHICLE_LENGTH + min_gap) * i

        decrement = 0
        edge_indx = 0
        startpositions = []
        startlanes = []
        for i in range(self.vehicles.num_vehicles):
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

        return startpositions, startlanes

    def gen_custom_start_pos(self, initial_config, **kwargs):
        """
        Generates a user defined set of starting postions. Optional

        Parameters
        ----------
        initial_config: InitialConfig type
            see flow/core/params.py
        kwargs: dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions: list
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes: list
            list of start lanes
        """
        raise NotImplementedError

    def _get_start_pos_util(self, initial_config, **kwargs):
        """
        Performs some pre-processing to the initial_config and **kwargs terms,
        and returns the necessary values for all starting position generating
        functions.

        Parameters
        ----------
        initial_config: InitialConfig type
            see flow/core/params.py
        kwargs: dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        x0: float
            starting position of the first vehicle, in meters
        min_gap: float
            minimum gap between vehicles
        bunching: float
            the amount of space freed up in the network (per lane)
        lanes_distribution: int
            number of lanes the vehicles are supposed to be distributed over
        available_length: float
            total available free space for vehicle to be placed, over all lanes
            within the distributable lanes, in meters
        initial_config: InitialConfig type
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
            max_lane = max([self.num_lanes(edge_id)
                            for edge_id in self.get_edge_list()])
        else:
            max_lane = max([self.num_lanes(edge_id)
                            for edge_id in initial_config.edges_distribution])

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
        num_vehicles = self.vehicles.num_vehicles

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

        return x0, min_gap, bunching, lanes_distribution, available_length, \
            available_edges, initial_config

    def edge_length(self, edge_id):
        """
        Returns the length of a given edge.
        """
        if ":" in edge_id:  # junctions have no length in this sense
            return 0
        return self.edges[edge_id]["length"]

    def speed_limit(self, edge_id):
        """
        Returns the speed limit of a given edge.
        """
        if ":" in edge_id:  # give vehicles in junctions a default speed limit
            return 30
        return self.edges[edge_id]["speed"]

    def num_lanes(self, edge_id):
        """
        Returns the number of lanes of a given edge.
        """
        if ":" in edge_id:  # treat all junctions as single lane
            return 1
        return self.edges[edge_id]["lanes"]

    def get_edge_list(self):
        """
        Returns the name of all edges in the network.
        """
        return list(self.edges.keys())

    def __str__(self):
        # TODO(cathywu) return the parameters too.
        return "Scenario " + self.name + " with " + \
               str(self.vehicles.num_vehicles) + " vehicles."
