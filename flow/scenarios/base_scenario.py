import logging
import numpy as np
import math
from collections import OrderedDict

from rllab.core.serializable import Serializable

from flow.core.generator import Generator
from flow.core.params import InitialConfig
from flow.controllers.rlcarfollowingcontroller import RLCarFollowingController


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

        Raises
        ------
        ValueError
            If no "length" is provided in net_params
        """
        Serializable.quick_init(self, locals())

        self.name = name
        self.generator_class = generator_class
        self.vehicles = vehicles
        self.net_params = net_params
        self.initial_config = initial_config

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
        # meant to be distributed (to be calculated during subclass __init__(),
        # or specified in net_params)
        if not hasattr(self, "length"):
            if "length" in self.net_params.additional_params:
                self.length = self.net_params.additional_params["length"]
            else:
                raise ValueError("The network does not have a specified length.")

        # generate starting position for vehicles in the network
        if self.initial_config.positions is None:
            self.initial_config.positions, self.initial_config.lanes = \
                self.generate_starting_positions()

        self.cfg = self.generate()

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
        if self.initial_config.spacing == "gaussian_additive":
            startpositions, startlanes = \
                self.gen_gaussian_additive_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "uniform_random":
            startpositions, startlanes = \
                self.gen_uniform_random_spacing(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "gaussian":
            startpositions, startlanes = \
                self.gen_gaussian_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "uniform":
            startpositions, startlanes = \
                self.gen_even_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "uniform_in_lane":
            startpositions, startlanes = \
                self.get_uniform_in_lane(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "custom":
            startpositions, startlanes = \
                self.gen_custom_start_pos(self.initial_config, **kwargs)
        elif self.initial_config.spacing == "gen_no_overlap_lane_starts":
            startpositions, startlanes = \
                self.gen_no_overlap_lane_starts(self.initial_config, **kwargs)
        else:
            raise ValueError('"spacing" argument in initial_config does not contain a valid option')

        return startpositions, startlanes


    def gen_uniform_random_spacing(self, initial_config, **kwargs):
        """
        Generate random start positions via uniform random distribution.
        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.

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
        increment = (distribution_length - bunching) * \
            initial_config.lanes_distribution / self.vehicles.num_vehicles

        x = np.random.uniform(0, x0, initial_config.lanes_distribution)
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
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        # perturb from uniform distribution
        for i in range(len(x_start)):
            perturb = np.random.uniform(-initial_config.scale, initial_config.scale, None)
            x_start[i] = (x_start[i] + perturb) % distribution_length

            pos = self.get_edge(x_start[i])

            # ensures that vehicles are not placed in an internal junction
            if pos[0] in dict(self.internal_edgestarts).keys():
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

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            startpositions.append(pos)

        return startpositions, startlanes

    def gen_even_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are uniformly spaced across the network.

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
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
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

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            if initial_config.distribution_length > self.length:
                logging.warning('"distribution_length" cannot be larger than '
                                'the length of network; setting to max value')
            else:
                distribution_length = initial_config.distribution_length

        if initial_config.lanes_distribution > self.lanes:
            logging.warning('"lanes_distribution" is greater than the number '
                            'of lanes in the network; distributing over all '
                            'lanes instead.')
            lanes_distribution = self.lanes
        elif initial_config.lanes_distribution < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        else:
            lanes_distribution = initial_config.lanes_distribution

        startpositions = []
        startlanes = []
        increment = (distribution_length - bunching) / np.ceil(
            self.vehicles.num_vehicles / lanes_distribution)

        if increment < 5:  # 5 is the length of all vehicles
            logging.warning("distribution is too compact; replacing with tight"
                            " (zero headway) starting positions")
            increment = 5

        x = [x0] * lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

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

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be
            # distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        return startpositions, startlanes


    def get_uniform_in_lane(self, initial_config, **kwargs):
        """
        Generates start positions that are uniformly spaced across the network.

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
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
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

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            if initial_config.distribution_length > self.length:
                logging.warning('"distribution_length" cannot be larger than '
                                'the length of network; setting to max value')
            else:
                distribution_length = initial_config.distribution_length

        if initial_config.lanes_distribution > self.lanes:
            logging.warning('"lanes_distribution" is greater than the number '
                            'of lanes in the network; distributing over all '
                            'lanes instead.')
            lanes_distribution = self.lanes
        elif initial_config.lanes_distribution - initial_config.starting_lane  < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        elif initial_config.starting_lane > self.lanes:
            logging.warning('starting lane is too high!')
        else:
            lanes_distribution = initial_config.lanes_distribution - initial_config.starting_lane

        startpositions = []
        startlanes = []

        cars_per_lane = [0] * initial_config.starting_lane + [math.floor(self.vehicles.num_vehicles / lanes_distribution)] * lanes_distribution


        i = 0
        while sum(cars_per_lane) < self.vehicles.num_vehicles:
            cars_per_lane[i] += 1
            i += 1
        car_distributions = [np.random.choice(int((distribution_length - bunching) / 8), num_cars, replace=False).tolist() for num_cars in cars_per_lane]
        car_distributions = [[x * 8 + x0 for x in lane_dist] for lane_dist in car_distributions]

        for lane, lane_poses in enumerate(car_distributions):
            for car_pos in lane_poses:
                pos = self.get_edge(car_pos)
                startpositions.append(pos)
                startlanes.append(lane)


        return startpositions, startlanes

    def gen_gaussian_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are perturbed from a uniformly spaced
        distribution by some gaussian noise.

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
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
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

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            if initial_config.distribution_length > self.length:
                logging.warning('"distribution_length" cannot be larger than '
                                'the length of network; setting to max value')
            else:
                distribution_length = initial_config.distribution_length

        if initial_config.lanes_distribution > self.lanes:
            logging.warning('"lanes_distribution" is greater than the number '
                            'of lanes in the network; distributing over all '
                            'lanes instead.')
            lanes_distribution = self.lanes
        elif initial_config.lanes_distribution < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        else:
            lanes_distribution = initial_config.lanes_distribution

        startpositions = []
        startlanes = []
        increment = (distribution_length - bunching) / np.ceil(
            self.vehicles.num_vehicles / lanes_distribution)

        # if the increment is too small, bunch vehicles as close together as
        # possible
        if increment < 5:  # 5 is the length of all vehicles
            return self.gen_even_start_pos(initial_config, **kwargs)

        x = [x0] * lanes_distribution
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
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        # perturb from uniform distribution
        for i in range(len(x_start)):
            perturb = np.random.normal(loc=0, scale=initial_config.scale)
            x_start[i] = (x_start[i] + perturb) % distribution_length

            pos = self.get_edge(x_start[i])

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

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            startpositions.append(pos)

        return startpositions, startlanes

    def gen_gaussian_additive_start_pos(self, initial_config, **kwargs):
        """
        Generate random start positions via additive Gaussian.
        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.

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
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
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

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            if initial_config.distribution_length > self.length:
                logging.warning('"distribution_length" cannot be larger than '
                                'the length of network; setting to max value')
            else:
                distribution_length = initial_config.distribution_length

        if initial_config.lanes_distribution > self.lanes:
            logging.warning('"lanes_distribution" is greater than the number '
                            'of lanes in the network; distributing over all '
                            'lanes instead.')
            lanes_distribution = self.lanes
        elif initial_config.lanes_distribution < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        else:
            lanes_distribution = initial_config.lanes_distribution

        startpositions = []
        startlanes = []
        mean = (distribution_length - bunching) / np.ceil(
            self.vehicles.num_vehicles / lanes_distribution)

        # if the mean (increment) is too small, bunch vehicles as close together
        # as possible
        if mean < 5:  # 5 is the length of all vehicles
            return self.gen_even_start_pos(initial_config, **kwargs)

        x = [x0] * lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:

            pos = self.get_edge(x[lane_count])

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

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            # collect the position and lane number of each new vehicle
            startpositions.append(pos)
            startlanes.append(lane_count)

            # calculate the increment given the mean, and ensure that the
            # increment is never too large or too small (between 0 and the
            # length of the network)
            increment = np.clip(
                np.random.normal(scale=mean/initial_config.downscale, loc=mean),
                a_min=0, a_max=self.length
            )

            x[lane_count] = (x[lane_count] + increment) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_no_overlap_lane_starts(self, initial_config, **kwargs):
        """
        Generate random start positions via additive Gaussian.
        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.

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
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
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

        distribution_length = self.length
        if initial_config.distribution_length is not None:
            if initial_config.distribution_length > self.length:
                logging.warning('"distribution_length" cannot be larger than '
                                'the length of network; setting to max value')
            else:
                distribution_length = initial_config.distribution_length

        if initial_config.lanes_distribution > self.lanes:
            logging.warning('"lanes_distribution" is greater than the number '
                            'of lanes in the network; distributing over all '
                            'lanes instead.')
            lanes_distribution = self.lanes
        elif initial_config.lanes_distribution < 1:
            logging.warning('"lanes_distribution" is too small; setting to 1')
            lanes_distribution = 1
        else:
            lanes_distribution = initial_config.lanes_distribution

        startpositions = []
        startlanes = []
        mean = (distribution_length - bunching) / np.ceil(
            self.vehicles.num_vehicles / lanes_distribution)

        # if the mean (increment) is too small, bunch vehicles as close together
        # as possible
        if mean < 5:  # 5 is the length of all vehicles
            return self.gen_even_start_pos(initial_config, **kwargs)

        x = x0
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:

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

            # collect the position and lane number of each new vehicle
            startpositions.append(pos)
            startlanes.append(lane_count)

            # calculate the increment given the mean, and ensure that the
            # increment is never too large or too small (between 0 and the
            # length of the network)
            increment = np.clip(
                np.random.normal(scale=mean/initial_config.downscale, loc=mean),
                a_min=0, a_max=self.length
            )

            x = (x + increment) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

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

    def __str__(self):
        # TODO(cathywu) return the parameters too.
        return "Scenario " + self.name + " with " + str(self.vehicles.num_vehicles) + " vehicles."


