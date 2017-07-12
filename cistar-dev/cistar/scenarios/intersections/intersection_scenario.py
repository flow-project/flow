import numpy as np
import random
from cistar.core.scenario import Scenario
from cistar.scenarios.intersections.gen import *
import pdb


class TwoWayIntersectionScenario(Scenario):

    def __init__(self, name, type_params, net_params, cfg_params=None, initial_config=None, cfg=None):
        """
        Initializes a two-way intersection scenario. Required net_params: horizontal_length_before,
        horizontal_length_after, horizontal_lanes, vertical_length_before, vertical_length_after, vertical_lanes,
        speed_limit. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        self.left_len = net_params["horizontal_length_in"]
        self.right_len = net_params["horizontal_length_out"]
        self.bottom_len = net_params["vertical_length_in"]
        self.top_len = net_params["vertical_length_out"]

        self.horizontal_junction_len = 2.9 + 3.3 * net_params["vertical_lanes"]
        self.vertical_junction_len = 2.9 + 3.3 * net_params["horizontal_lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params["length"] = self.left_len + self.right_len + self.horizontal_junction_len + \
            self.bottom_len + self.top_len + self.vertical_junction_len

        super().__init__(name, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config, cfg=cfg,
                         generator_class=TwoWayIntersectionGenerator)

        self.length = self.net_params["length"]

        if "horizontal_lanes" not in self.net_params:
            raise ValueError("number of horizontal lanes not supplied")

        if "vertical_lanes" not in self.net_params:
            raise ValueError("number of vertical lanes not supplied")

        self.lanes = {"top": self.net_params["vertical_lanes"], "bottom": self.net_params["vertical_lanes"],
                      "left": self.net_params["horizontal_lanes"], "right": self.net_params["horizontal_lanes"]}

        # enter_lane specifies which lane a car enters given a certain direction
        self.enter_lane = {"horizontal": "left", "vertical": "bottom"}

        if "speed_limit" not in self.net_params:
            raise ValueError("speed limit not supplied")

        # if the speed limit is a single number, then all lanes have the same speed limit
        if isinstance(self.net_params["speed_limit"], int) or isinstance(self.net_params["speed_limit"], float):
            self.speed_limit = {"horizontal": self.net_params["speed_limit"],
                                "vertical": self.net_params["speed_limit"]}
        # if the speed limit is a dict with separate values for vertical and horizontal,
        # then they are set as such
        elif "vertical" in self.net_params["speed_limit"] and "horizontal" in self.net_params["speed_limit"]:
            self.speed_limit = {"horizontal": self.net_params["speed_limit"]["horizontal"],
                                "vertical": self.net_params["speed_limit"]["vertical"]}
        else:
            raise ValueError('speed limit must contain a number or a dict with keys: "vertical" and "horizontal"')

        # defines edge starts for road sections
        self.edgestarts = \
            [("bottom", 0),
             ("top", self.bottom_len + self.vertical_junction_len),
             ("left", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len)),
             ("right", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len) + self.left_len + self.horizontal_junction_len)]

        # defines edge starts for intersections
        self.intersection_edgestarts = \
            [(":center_%s" % (1+self.lanes["left"]), self.bottom_len),
             (":center_1", 1000 * (self.bottom_len + self.vertical_junction_len + self.top_len) + self.left_len)]

        if "positions" not in self.initial_config:
            self.initial_config["positions"] = self.generate_starting_positions()

        if "shuffle" not in self.initial_config:
            self.initial_config["shuffle"] = False
        if not cfg:
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
        Given an edge name and relative position, return the absolute position on the track.
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

    def generate_starting_positions(self, x0=1):
        """
        Generates starting positions for vehicles in the network
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []

        bunch_factor = 0
        if "bunching" in self.initial_config:
            bunch_factor = self.initial_config["bunching"]

        if "spacing" in self.initial_config:
            if self.initial_config["spacing"] == "gaussian":
                downscale = 5
                if "downscale" in self.initial_config:
                    downscale = self.initial_config["downscale"]
                startpositions = self.gen_random_start_pos(downscale, bunch_factor)
            if self.initial_config["spacing"] == "edge_start":
                startpositions = self.gen_random_end_pos(self.initial_config["intensity"],
                                                        self.initial_config["enter_speed"])
        else:
            startpositions = self.gen_even_start_positions(bunch_factor)

        return startpositions

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
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        startpositions = []
        mean = (self.length - self.horizontal_junction_len - self.vertical_junction_len - bunching) / self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            pos = self.get_edge(x)

            # ensures that vehicles are not placed in the intersection
            for center_tuple in self.intersection_edgestarts:
                if center_tuple[0] in pos[0]:
                    x += self.vertical_junction_len
                    pos = self.get_edge(x)

            startpositions.append(pos)
            x += np.random.normal(scale=mean / downscale, loc=mean)

        return startpositions

    def gen_random_end_pos(self, rate, v_enter):
        """
        Generate random positions starting from the ends of the track.
        Vehicles are spaced so that no car can arrive at the 
        control portion of the track more often than...
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]    
        """
        start_positions = []
        x = 1
        while len(start_positions) < self.num_vehicles:
            left_lane = np.random.randint(2, size=1)
            d_inc = v_enter*random.expovariate(1.0/rate)
            # FIX to get length of car that has been placed already
            if d_inc > 5:
                x += d_inc
                if left_lane:
                    start_positions.append(("left", x))
                else:
                    start_positions.append(("bottom", x))
        return start_positions

