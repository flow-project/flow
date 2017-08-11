import numpy as np

from cistar.core.scenario import Scenario
from cistar.scenarios.braess_paradox.gen import *


class BraessParadoxScenario(Scenario):

    def __init__(self, name, type_params, net_params, cfg_params=None, initial_config=None, cfg=None):
        """
        Initializes a Braess Paradox scenario.
        Required net_params: angle, edge_length, lanes, resolution speed_limit.
        """
        self.angle = net_params["angle"]
        self.edge_len = net_params["edge_length"]
        self.curve_len = (0.75 * self.edge_len * np.sin(self.angle)) * np.pi
        self.junction_len = 2.9 + 3.3 * net_params["lanes"]
        self.inner_space_len = 0.28
        self.horz_len = 2 * self.edge_len * np.cos(self.angle)

        # instantiate "length" in net params
        # net_params["length"] = 4 * self.edge_len + 4 * self.junction_len + 2 * self.curve_len + self.horz_len
        net_params["length"] = 4 * self.edge_len + 2 * self.curve_len + self.horz_len

        super().__init__(name, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config, cfg=cfg,
                         generator_class=BraessParadoxGenerator)

        if "length" not in self.net_params:
            raise ValueError("length of circle not supplied")
        self.length = self.net_params["length"]

        if "lanes" not in self.net_params:
            raise ValueError("lanes of circle not supplied")
        self.lanes = self.net_params["lanes"]

        if "resolution" not in self.net_params:
            raise ValueError("resolution of circle not supplied")
        self.resolution = self.net_params["resolution"]

        # number of lanes vehicles should be distributed in at the start of a rollout
        # must be less than or equal to the number of lanes in the network
        if "lanes_distribution" not in self.net_params:
            self.lanes_distribution = 1
        else:
            self.lanes_distribution = min(self.net_params["lanes_distribution"], self.lanes)

        # defines edge starts for road sections
        self.edgestarts = \
            [("B",   0),
             ("BA1", self.curve_len),
             ("BA2", self.curve_len + self.horz_len),
             ("AC",  2 * self.curve_len + self.horz_len),
             ("CB",  2 * self.curve_len + self.horz_len + self.edge_len),
             ("AD",  2 * self.curve_len + self.horz_len + 2 * self.edge_len),
             ("DB",  2 * self.curve_len + self.horz_len + 3 * self.edge_len),
             ("CD",  2 * self.curve_len + self.horz_len + 4 * self.edge_len)]

        # defines edge starts for intersections
        self.intersection_edgestarts = \
            [(":A", 2 * self.curve_len + self.horz_len + 2 * self.edge_len),
             (":B", 0),
             (":C", 2 * self.curve_len + self.horz_len),
             (":D", 2 * self.curve_len + self.horz_len + 3 * self.edge_len)]

        # generate starting position for vehicles in the network
        if "positions" not in self.initial_config:
            self.initial_config["positions"], self.initial_config["lanes"] = self.generate_starting_positions()

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
                startpositions = self.gen_random_start_positions(downscale, bunch_factor, x0=x0)
        else:
            startpositions = self.gen_even_start_positions(bunch_factor, x0=x0)

        return startpositions

    def gen_even_start_positions(self, bunching, x0=1):
        """
        Generate uniformly spaced start positions.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
        """
        distribution_len = 2 * self.curve_len + self.horz_len
        startpositions = []
        startlanes = []
        increment = (distribution_len - bunching) * self.lanes_distribution / self.num_vehicles

        x = [x0] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % self.length

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_random_start_positions(self, downscale=5, bunching=0, x0=1):
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
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])
            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + np.random.normal(scale=mean / downscale, loc=mean)) % self.length

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes
