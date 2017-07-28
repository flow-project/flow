import numpy as np

from cistar.core.scenario import Scenario
from cistar.scenarios.figure8.gen import Figure8Generator


class Figure8Scenario(Scenario):
    def __init__(self, name, type_params, net_params, cfg_params=None,
                 initial_config=None, cfg=None):
        """
        Initializes a figure 8 scenario. Required net_params: radius_ring, lanes,
        speed_limit, resolution. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        self.ring_edgelen = net_params["radius_ring"] * np.pi / 2.
        self.intersection_len = 2 * net_params["radius_ring"]
        self.junction_len = 2.9 + 3.3 * net_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params["length"] = 6 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + \
            10 * self.inner_space_len

        super().__init__(name, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config, cfg=cfg,
                         generator_class=Figure8Generator)

        if "radius_ring" not in self.net_params:
            raise ValueError("radius of ring not supplied")
        self.radius_ring = self.net_params["radius_ring"]

        self.length = self.net_params["length"]

        if "lanes" not in self.net_params:
            raise ValueError("number of lanes not supplied")
        self.lanes = self.net_params["lanes"]

        if "speed_limit" not in self.net_params:
            raise ValueError("speed limit not supplied")
        self.speed_limit = self.net_params["speed_limit"]

        if "resolution" not in self.net_params:
            raise ValueError("resolution of circular sections not supplied")
        self.resolution = self.net_params["resolution"]

        # number of lanes vehicles should be distributed in at the start of a rollout
        # must be less than or equal to the number of lanes in the network
        if "lanes_distribution" not in self.net_params:
            self.lanes_distribution = 1
        else:
            self.lanes_distribution = min(self.net_params["lanes_distribution"], self.lanes)

        # defines edge starts for road sections
        self.edgestarts = [("bottom_lower_ring", 0 + self.inner_space_len),
                           ("right_lower_ring_in", self.ring_edgelen + 2 * self.inner_space_len),
                           ("right_lower_ring_out", self.ring_edgelen + self.intersection_len / 2 + self.junction_len + 3 * self.inner_space_len),
                           ("left_upper_ring", self.ring_edgelen + self.intersection_len + self.junction_len + 4 * self.inner_space_len),
                           ("top_upper_ring", 2 * self.ring_edgelen + self.intersection_len + self.junction_len + 5 * self.inner_space_len),
                           ("right_upper_ring", 3 * self.ring_edgelen + self.intersection_len + self.junction_len + 6 * self.inner_space_len),
                           ("bottom_upper_ring_in", 4 * self.ring_edgelen + self.intersection_len + self.junction_len + 7 * self.inner_space_len),
                           ("bottom_upper_ring_out", 4 * self.ring_edgelen + 3 / 2 * self.intersection_len + 2 * self.junction_len + 8 * self.inner_space_len),
                           ("top_lower_ring", 4 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 9 * self.inner_space_len),
                           ("left_lower_ring", 5 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 10 * self.inner_space_len)]

        self.edgepos = {"bottom_lower_ring": 0 + self.inner_space_len,
                        "right_lower_ring_in": self.ring_edgelen + 2 * self.inner_space_len,
                        "right_lower_ring_out": self.ring_edgelen + self.intersection_len / 2 + self.junction_len + 3 * self.inner_space_len,
                        "left_upper_ring": self.ring_edgelen + self.intersection_len + self.junction_len + 4 * self.inner_space_len,
                        "top_upper_ring": 2 * self.ring_edgelen + self.intersection_len + self.junction_len + 5 * self.inner_space_len,
                        "right_upper_ring": 3 * self.ring_edgelen + self.intersection_len + self.junction_len + 6 * self.inner_space_len,
                        "bottom_upper_ring_in": 4 * self.ring_edgelen + self.intersection_len + self.junction_len + 7 * self.inner_space_len,
                        "bottom_upper_ring_out": 4 * self.ring_edgelen + 3 / 2 * self.intersection_len + 2 * self.junction_len + 8 * self.inner_space_len,
                        "top_lower_ring": 4 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 9 * self.inner_space_len,
                        "left_lower_ring": 5 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 10 * self.inner_space_len}

        # defines edge starts for intersections
        self.intersection_edgestarts = \
            [(":center_intersection_%s" % (1+self.lanes), self.ring_edgelen + self.intersection_len / 2 + 3 * self.inner_space_len),
             (":center_intersection_1", 4 * self.ring_edgelen + 3 / 2 * self.intersection_len + self.junction_len + 8 * self.inner_space_len)]

        self.intersection_pos = \
            {":center_intersection_%s" % (1+self.lanes): self.ring_edgelen + self.intersection_len / 2 + 3 * self.inner_space_len,
             ":center_intersection_1": 4 * self.ring_edgelen + 3 / 2 * self.intersection_len + self.junction_len + 8 * self.inner_space_len}

        self.extra_edgestarts = \
            [("bottom_lower_ring", 0),
             ("right_lower_ring_in", self.ring_edgelen + self.inner_space_len),
             ("right_lower_ring_out",
              self.ring_edgelen + self.intersection_len / 2 + self.junction_len + 2 * self.inner_space_len),
             ("left_upper_ring",
              self.ring_edgelen + self.intersection_len + self.junction_len + 3 * self.inner_space_len),
             ("top_upper_ring",
              2 * self.ring_edgelen + self.intersection_len + self.junction_len + 4 * self.inner_space_len),
             ("right_upper_ring",
              3 * self.ring_edgelen + self.intersection_len + self.junction_len + 5 * self.inner_space_len),
             ("bottom_upper_ring_in",
              4 * self.ring_edgelen + self.intersection_len + self.junction_len + 6 * self.inner_space_len),
             ("bottom_upper_ring_out",
              4 * self.ring_edgelen + 3 / 2 * self.intersection_len + 2 * self.junction_len + 7 * self.inner_space_len),
             ("top_lower_ring",
              4 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 8 * self.inner_space_len),
             ("left_lower_ring",
              5 * self.ring_edgelen + 2 * self.intersection_len + 2 * self.junction_len + 9 * self.inner_space_len)]

        # generate starting position for vehicles in the network
        if "positions" not in self.initial_config:
            self.initial_config["positions"], self.initial_config["lanes"] = self.generate_starting_positions()

        if "shuffle" not in self.initial_config:
            self.initial_config["shuffle"] = False
        if not cfg:
            # FIXME(cathywu) Resolve this inconsistency. Base class does not
            # call generate, but child class does. What is the convention?
            self.cfg = self.generate()

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
        for (e, s) in self.edgestarts:
            if x >= s:
                starte = e
                startx = x - s
        return starte, startx

    def get_x(self, edge, position):
        """
        Given an edge name and relative position, return the absolute
        position on the track.
        :param edge: name of edge (string)
        :param position: relative position on edge
        :return: absolute position of the vehicle on the track given a reference (origin)
        """
        # check it the vehicle is in a lane
        if edge in self.edgepos.keys():
            return position + self.edgepos[edge]

        # if the vehicle is not in a lane, check if it is on an intersection
        if edge in self.intersection_pos.keys():
            return position + self.intersection_pos[edge]

        # finally, check if it is in the connection between lanes
        for extra_tuple in self.extra_edgestarts:
            if extra_tuple[0] in edge:
                edgestart = extra_tuple[1]
                return position + edgestart

    def generate_starting_positions(self, x0=1):
        """
        Generates starting positions for vehicles in the network
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        startlanes = []

        bunch_factor = 0
        if "bunching" in self.initial_config:
            bunch_factor = self.initial_config["bunching"]

        if "spacing" in self.initial_config:
            if self.initial_config["spacing"] == "gaussian":
                downscale = 5
                if "downscale" in self.initial_config:
                    downscale = self.initial_config["downscale"]
                startpositions, startlanes = self.gen_random_start_pos(downscale, bunch_factor, x0=x0)
        else:
            startpositions, startlanes = self.gen_even_start_positions(bunch_factor, x0=x0)

        return startpositions, startlanes

    def gen_even_start_positions(self, bunching, x0=1):
        """
        Generate uniformly spaced start positions.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        startlanes = []
        increment = (self.length - bunching) * self.lanes_distribution / self.num_vehicles

        x = [x0] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_vehicles:

            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in the intersection
            for center_tuple in self.intersection_edgestarts:
                if center_tuple[0] in pos[0]:
                    x[lane_count] += self.junction_len
                    pos = self.get_edge(x[lane_count])

            # collect the position and lane number of each new vehicle
            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_random_start_pos(self, downscale=5, bunching=0, x0=1):
        """
        Generate random start positions via additive Gaussian.

        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        startlanes = []
        mean = (self.length - bunching) * self.lanes_distribution / self.num_vehicles

        x = [x0] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_vehicles:

            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in the intersection
            for center_tuple in self.intersection_edgestarts:
                if center_tuple[0] in pos[0]:
                    x[lane_count] += self.junction_len
                    pos = self.get_edge(x[lane_count])

            # collect the position and lane number of each new vehicle
            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + np.random.normal(scale=mean / downscale, loc=mean)) % self.length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes
