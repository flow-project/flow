import numpy as np

from cistar.core.scenario import Scenario
from cistar.scenarios.figure8.gen import Figure8Generator


class Figure8Scenario(Scenario):
    def __init__(self, name, type_params, net_params, cfg_params=None,
                 initial_config=None, cfg=None):
        """
        Initializes a loop scenario. Required net_params: length, lanes,
        speed_limit, resolution. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        super().__init__(name, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config, cfg=cfg,
                         generator_class=Figure8Generator)

        if "radius_ring" not in self.net_params:
            raise ValueError("radius of ring not supplied")
        self.radius_ring = self.net_params["radius_ring"]
        self.length = self.radius_ring * (3 * np.pi + 4)

        if "lanes" not in self.net_params:
            raise ValueError("number of lanes not supplied")
        self.lanes = self.net_params["lanes"]

        if "speed_limit" not in self.net_params:
            raise ValueError("speed limit not supplied")
        self.speed_limit = self.net_params["speed_limit"]

        if "priority" not in self.net_params:
            raise ValueError("intersection priority not supplied")
        elif self.net_params["priority"] != "None" and self.net_params["priority"] != "top_bottom" and \
                        self.net_params["priority"] != "left_right":
            raise ValueError('priority must be "top_bottom", "left_right", or "None" ')

        if "resolution" not in self.net_params:
            raise ValueError("resolution of circle not supplied")
        self.resolution = self.net_params["resolution"]

        ring_edgelen = self.radius_ring * np.pi / 2.
        intersection_edgelen = 2 * self.radius_ring

        self.edgestarts = [("bottom_lower_ring", 0),
                           ("right_lower_ring_in", ring_edgelen),
                           ("right_lower_ring_out", ring_edgelen + intersection_edgelen / 2),
                           ("left_upper_ring", ring_edgelen + intersection_edgelen),
                           ("top_upper_ring", 2 * ring_edgelen + intersection_edgelen),
                           ("right_upper_ring", 3 * ring_edgelen + intersection_edgelen),
                           ("bottom_upper_ring_in", 4 * ring_edgelen + intersection_edgelen),
                           ("bottom_upper_ring_out", 4 * ring_edgelen + 3 / 2 * intersection_edgelen),
                           ("top_lower_ring", 4 * ring_edgelen + 2 * intersection_edgelen),
                           ("left_lower_ring", 5 * ring_edgelen + 2 * intersection_edgelen)]

        self.junctionstarts = [(":right_lower_ring", ring_edgelen),
                               (":center_intersection_2", ring_edgelen + intersection_edgelen / 2),
                               ("bottom_upper_ring", 4 * ring_edgelen + intersection_edgelen),
                               (":center_intersection_1", 4 * ring_edgelen + 3 / 2 * intersection_edgelen)]

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
        :return:
        """
        # check it the vehicle is in a lane
        for edge_tuple in self.edgestarts:
            if edge_tuple[0] in edge:
                edgestart = edge_tuple[1]
                return position + edgestart

        # if the vehicle is not in a lane, check if it is on a junction
        for junction_tuple in self.junctionstarts:
            if junction_tuple[0] in edge:
                edgestart = junction_tuple[1]
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
        mean = (self.length - bunching) / self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            pos = self.get_edge(x)
            startpositions.append(pos)
            x += np.random.normal(scale=mean / downscale, loc=mean)

        return startpositions
