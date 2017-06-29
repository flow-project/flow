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
        print(net_params["length"])

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

        # defines edge starts for intersections
        self.intersection_edgestarts = \
            [(":center_intersection_%s" % (1+self.lanes), self.ring_edgelen + self.intersection_len / 2 + 3 * self.inner_space_len),
             (":center_intersection_1", 4 * self.ring_edgelen + 3 / 2 * self.intersection_len + self.junction_len + 8 * self.inner_space_len)]

        print(self.intersection_edgestarts)
        print(self.junction_len)

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

        if "positions" not in self.initial_config:
            bunch_factor = 0
            if "bunching" in self.initial_config:
                bunch_factor = self.initial_config["bunching"]

            if "spacing" in self.initial_config:
                if self.initial_config["spacing"] == "gaussian":
                    downscale = 5
                    if "downscale" in self.initial_config:
                        downscale = self.initial_config["downscale"]
                    self.initial_config["positions"] = self.gen_random_start_pos(downscale, bunch_factor)
            else:
                self.initial_config["positions"] = self.gen_even_start_positions(bunch_factor)

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
        for edge_tuple in self.edgestarts:
            if edge_tuple[0] == edge:
                edgestart = edge_tuple[1]
                return position + edgestart

        # if the vehicle is not in a lane, check if it is on an intersection
        for center_tuple in self.intersection_edgestarts:
            if center_tuple[0] in edge:
                edgestart = center_tuple[1]
                return position + edgestart

        # finally, check if it is in the connection between lanes
        for extra_tuple in self.extra_edgestarts:
            if extra_tuple[0] in edge:
                edgestart = extra_tuple[1]
                return position + edgestart

    def gen_even_start_positions(self, bunching):
        """
        Generate uniformly spaced start positions.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        increment = (self.length - bunching) / self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            # pos is a tuple (route, departPos)
            pos = self.get_edge(x)
            startpositions.append(pos)
            x += increment

        return startpositions

    def gen_random_start_pos(self, downscale=5, bunching=0):
        """
        Generate random start positions via additive Gaussian.

        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        mean = (self.length - 2 * self.junction_len - bunching) / self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            pos = self.get_edge(x)

            # ensures that vehicles are not placed in the intersection
            for center_tuple in self.intersection_edgestarts:
                if center_tuple[0] in pos[0]:
                    x += self.junction_len
                    pos = self.get_edge(x)

            startpositions.append(pos)
            x += np.random.normal(scale=mean / downscale, loc=mean)

        return startpositions
