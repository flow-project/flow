import logging
import numpy as np
from collections import OrderedDict

from rllab.core.serializable import Serializable

from cistar.core.generator import Generator
from cistar.core.params import InitialConfig
from cistar.controllers.rlcontroller import RLController


class Scenario(Serializable):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig()):
        """
        Abstract base class. Initializes a new scenario. This class can be
        instantiated once and reused in multiple experiments. Note that this
        function stores all the relevant parameters. The generate() function
        still needs to be called separately.

        :param name: A tag associated with the scenario
        :param generator_class: Class for generating a configuration files
        and net files with placed vehicles, e.g. CircleGenerator
        :param vehicles: See README.md
        :param net_params: See README.md
        :param cfg_params: See README.md
        :param initial_config:
            { 'shuffle' : True }: Shuffle the starting positions of vehicles.
            { 'positions' : [(route0, pos0), (route1, pos1), ... ]}: Places
            vehicles on route route0, position on edge (pos0).
            Note: this needs to be implemented in a child class.
        """
        Serializable.quick_init(self, locals())

        self.name = name
        self.generator_class = generator_class
        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config

        # parameters to be specified under each unique subclass's __init__() function
        self.edgestarts = self.specify_edge_starts()

        # these optional parameters need only be used if "no-internal-links" is set to "false"
        # while calling sumo's netconvert function
        self.internal_edgestarts = self.specify_internal_edge_starts()
        self.intersection_edgestarts = self.specify_intersection_edge_starts()

        # in case the user did not write the intersection edge-starts in internal edge-starts as
        # well (because of redundancy), merge the two together
        self.internal_edgestarts += self.intersection_edgestarts
        seen = set()
        self.internal_edgestarts = [item for item in self.internal_edgestarts
                                    if item[1] not in seen and not seen.add(item[1])]

        # total_edgestarts and total_edgestarts_dict contain all of the above edges, with the
        # former being ordered by position
        if self.net_params.no_internal_links:
            self.total_edgestarts = self.edgestarts
        else:
            self.total_edgestarts = self.edgestarts + self.internal_edgestarts
        self.total_edgestarts.sort(key=lambda tup: tup[1])

        self.total_edgestarts_dict = dict(self.total_edgestarts)

        # length of the network, or the portion of the network in which cars are meant to be distributed
        # (to be calculated during subclass __init__(), or specified in net_params)
        if not hasattr(self, "length"):
            if "length" in self.net_params.additional_params:
                self.length = self.net_params.additional_params["length"]
            else:
                raise ValueError("The network does not have a characteristic length specified.")

        # generate starting position for vehicles in the network
        if self.initial_config.positions is None:
            self.initial_config.positions, self.initial_config.lanes = self.generate_starting_positions()

        self.cfg = self.generate()

    def specify_edge_starts(self):
        """
        Defines edge starts for road sections w.r.t. some global reference frame.

        :return: a list of edge names and starting positions [(edge0, pos0), (edge1, pos1), ...]
        """
        raise NotImplementedError

    def specify_intersection_edge_starts(self):
        """
        Defines edge starts for intersections w.r.t. some global reference frame. Need not
        be specified if no intersections exist.
        These values can be used to determine the distance of some agent from the nearest
        and/or all intersections.

        :return: a list of intersection names and starting positions
                 [(intersection0, pos0), (intersection1, pos1), ...]
        """
        return []

    def specify_internal_edge_starts(self):
        """
        Defines the edge starts for internal edge nodes (caused by finite-length
        connections between road sections) w.r.t. some global reference frame.
        Need not be specified if "no-internal-links" is not specified or set to
        True in net_params.

        :return: a list of internal junction names and starting positions
                 [(internal0, pos0), (internal1, pos1), ...]
        """
        return []

    def generate(self):
        """
        Applies self.generator_class to create a net and corresponding cfg
        files, including placement of vehicles (name.rou.xml).
        :return: (path to cfg files (the .sumo.cfg), path to output files {
        "netstate", "amitran", "lanechange", "emission" })
        """
        logging.info("Config file not defined, generating using generator")

        # Default scenario parameters
        net_path = self.net_params.net_path
        cfg_path = self.net_params.cfg_path

        self.generator = self.generator_class(self.net_params, net_path, cfg_path, self.name)
        self.generator.generate_net(self.net_params)
        cfg_name = self.generator.generate_cfg(self.net_params)
        self.generator.make_routes(self, self.initial_config)

        return self.generator.cfg_path + cfg_name

    def get_edge(self, x):
        """
        Given an absolute position x on the track, returns the edge (name) and
        relative position on that edge.
        :param x: absolute position x
        :return: (edge (name, such as bottom, right, etc.), relative position on
        edge)
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
        Given an edge name and relative position, return the absolute position on the track.
        :param edge: name of edge (string)
        :param position: relative position on edge
        :return: absolute position of the vehicle on the track given a reference (origin)
        """
        if edge in dict(self.edgestarts).keys():
            return self.total_edgestarts_dict[edge] + position
        else:
            for edge_tuple in self.internal_edgestarts:
                if edge_tuple[0] in edge:
                    return edge_tuple[1] + position

    def generate_starting_positions(self, **kwargs):
        """
        Generates starting positions for vehicles in the network

        :param kwargs: additional arguments that may be updated beyond initial configurations,
                       such as modifying the starting position
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        if self.initial_config.spacing == "gaussian_additive":
            startpositions, startlanes = self.gen_gaussian_additive_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "gaussian":
            startpositions, startlanes = self.gen_gaussian_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "uniform":
            startpositions, startlanes = self.gen_even_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "custom":
            startpositions, startlanes = self.gen_custom_start_pos(self.initial_config, **kwargs)
        else:
            raise ValueError('"spacing" argument in initial_config does not contain a valid option')

        return startpositions, startlanes

    def gen_even_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are perturbed from a uniformly spaced distribution
        by some gaussian noise.

        :param kwargs:
        :param initial_config:
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = initial_config.bunching
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            distribution_length = initial_config.distribution_length

        startpositions = []
        startlanes = []
        increment = (distribution_length - bunching) * initial_config.lanes_distribution / self.vehicles.num_vehicles

        x = [x0] * initial_config.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in an internal junction
            if pos[0] in dict(self.internal_edgestarts).keys():
                indx_edge = 0
                for edge_tup in self.total_edgestarts:
                    if edge_tup[0] != pos[0]:
                        indx_edge += 1

                    # in case the internal edge is the farthest edge in the system,
                    # then the new vehicle should be placed at the start of the network,
                    # where a road network exists
                    if edge_tup == self.total_edgestarts[-1]:
                        if self.total_edgestarts[0][0] in dict(self.internal_edgestarts).keys():
                            indx_edge = 0
                        else:
                            indx_edge = -1

                next_edge_pos = self.total_edgestarts[indx_edge+1]
                x[lane_count] = next_edge_pos[1]
                pos = next_edge_pos

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be
            # distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_gaussian_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are perturbed from a uniformly spaced distribution
        by some gaussian noise.

        :param kwargs:
        :param initial_config:
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = initial_config.bunching
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            distribution_length = initial_config.distribution_length

        startpositions = []
        startlanes = []
        increment = (distribution_length - bunching) * initial_config.lanes_distribution / self.vehicles.num_vehicles

        x = [x0] * initial_config.lanes_distribution
        x_start = np.array([])
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:
            # collect the position and lane number of each new vehicle
            x_start = np.append(x_start, x[lane_count])
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % distribution_length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be
            # distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        # perturb from uniform distribution
        for i in range(len(x_start)):
            x_start[i] = \
                (x_start[i] + min(initial_config.scale,
                                  max(-initial_config.scale, np.random.normal(loc=0, scale=initial_config.scale)))) \
                % distribution_length

            pos = self.get_edge(x_start[i])

            # ensures that vehicles are not placed in an internal junction
            if pos[0] in dict(self.internal_edgestarts).keys():
                indx_edge = 0
                for edge_tup in self.total_edgestarts:
                    if edge_tup[0] != pos[0]:
                        indx_edge += 1

                    # in case the internal edge is the farthest edge in the system,
                    # then the new vehicle should be placed at the start of the network,
                    # where a road network exists
                    if edge_tup == self.total_edgestarts[-1]:
                        if self.total_edgestarts[0][0] in dict(self.internal_edgestarts).keys():
                            indx_edge = 0
                        else:
                            indx_edge = -1

                next_edge_pos = self.total_edgestarts[indx_edge+1]
                x[lane_count] = next_edge_pos[1]
                pos = next_edge_pos

            startpositions.append(pos)

        return startpositions, startlanes

    def gen_gaussian_additive_start_pos(self, initial_config, **kwargs):
        """
        Generate random start positions via additive Gaussian.

        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = initial_config.bunching
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            distribution_length = initial_config.distribution_length

        startpositions = []
        startlanes = []
        mean = (distribution_length - bunching) * initial_config.lanes_distribution / self.vehicles.num_vehicles

        x = [x0] * initial_config.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:

            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in an internal junction
            if pos[0] in dict(self.internal_edgestarts).keys():
                indx_edge = 0
                for edge_tup in self.total_edgestarts:
                    if edge_tup[0] != pos[0]:
                        indx_edge += 1

                    # in case the internal edge is the farthest edge in the system,
                    # then the new vehicle should be placed at the start of the network,
                    # where a road network exists
                    if edge_tup == self.total_edgestarts[-1]:
                        if self.total_edgestarts[0][0] in dict(self.internal_edgestarts).keys():
                            indx_edge = 0
                        else:
                            indx_edge = -1

                next_edge_pos = self.total_edgestarts[indx_edge+1]
                x[lane_count] = next_edge_pos[1]
                pos = next_edge_pos

            # collect the position and lane number of each new vehicle
            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = \
                (x[lane_count] + np.random.normal(scale=mean / initial_config.downscale, loc=mean)) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_upstream_start_pos(self, initial_config, **kwargs):
        """

        :param initial_config:
        :param kwargs:
        :return:
        """

    def gen_custom_start_pos(self, initial_config, **kwargs):
        raise NotImplementedError

    def __str__(self):
        # TODO(cathywu) return the parameters too.
        return "Scenario " + self.name + " with " + str(self.vehicles.num_vehicles) + " vehicles."
