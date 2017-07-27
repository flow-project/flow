import numpy as np

from cistar.core.scenario import Scenario
from cistar.scenarios.loop_merges.gen import *


class LoopMergesScenario(Scenario):

    def __init__(self, name, type_params, net_params, cfg_params=None, initial_config=None, cfg=None):
        """
        Initializes a two-way intersection scenario. Required net_params: horizontal_length_before,
        horizontal_length_after, horizontal_lanes, vertical_length_before, vertical_length_after, vertical_lanes,
        speed_limit. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        self.merge_in_len = net_params["merge_in_length"]
        self.merge_out_len = net_params["merge_out_length"]
        self.merge_in_angle = net_params["merge_in_angle"]
        self.merge_out_angle = net_params["merge_out_angle"]
        self.radius = net_params["ring_radius"]

        # the vehicles that start in the merging lane are distinguished by the presence of the string "merge"
        # in their names
        self.num_merge_vehicles = sum([x[1][0] for x in type_params.items() if "merge" in x[0]])

        ring_0_len = (self.merge_out_angle - self.merge_in_angle) % (2 * pi) * self.radius

        # TODO: find a good way of calculating these
        self.ring_0_0_len = 1.1 + 4 * net_params["lanes"]
        self.ring_1_0_len = 1.1 + 4 * net_params["lanes"]
        self.ring_0_n_len = 6.5
        self.ring_1_n_len = 6.5

        self.junction_len = 2.9 + 3.3 * net_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params["length"] = 2 * pi * self.radius + self.ring_0_n_len + self.ring_1_n_len

        super().__init__(name, type_params, net_params, cfg_params=cfg_params,
                         initial_config=initial_config, cfg=cfg,
                         generator_class=LoopMergesGenerator)

        if "length" not in self.net_params:
            raise ValueError("length of circle not supplied")
        self.length = self.net_params["length"]

        if "lanes" not in self.net_params:
            raise ValueError("lanes of circle not supplied")
        self.lanes = self.net_params["lanes"]

        if "speed_limit" not in self.net_params:
            raise ValueError("speed limit of circle not supplied")
        self.speed_limit = self.net_params["speed_limit"]

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
            [("ring_0", self.ring_0_n_len),
             ("ring_1", self.ring_0_n_len + ring_0_len + self.ring_1_n_len),
             ("merge_in", - self.merge_in_len - self.ring_0_0_len + self.ring_0_n_len),
             # ("merge_in", 500 * (2 * pi * self.radius)),
             ("merge_out", 1000 * (2 * pi * self.radius) + self.ring_1_0_len)]

        # defines edge starts for intersections
        # self.internal_edgestarts = \
        #     [(":ring_0_0", 0),
        #      (":ring_1_0", self.ring_0_0_len + ring_0_len),
        #      (":ring_0_%d" % self.lanes, - ring_0_n_len + self.ring_0_0_len),
        #      # (":ring_0_%d" % self.lanes, 500 * (2 * pi * self.radius) + self.merge_in_len),
        #      (":ring_1_%d" % self.lanes, 1000 * (2 * pi * self.radius))]

        self.internal_edgestarts = \
            [(":ring_0_%d" % self.lanes, 0),
             (":ring_1_%d" % self.lanes, self.ring_0_n_len + ring_0_len),
             (":ring_0_0", - self.ring_0_0_len + self.ring_0_n_len),
             (":ring_1_0", 1000 * (2 * pi * self.radius))]

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
        total_edgestarts = self.edgestarts + self.internal_edgestarts
        total_edgestarts.sort(key=lambda tup: tup[1])

        for (e, s) in total_edgestarts:
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
        for center_tuple in self.internal_edgestarts:
            if center_tuple[0] in edge:
                edgestart = center_tuple[1]
                return position + edgestart

    def generate_starting_positions(self, x0=1):
        """
        Generates starting positions for vehicles in the network
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        startpositions = []
        startlanes = []

        bunch_factor = 0
        if "bunching" in self.initial_config:
            bunch_factor = self.initial_config["bunching"]

        merge_bunch_factor = 0
        if "merge_bunching" in self.initial_config:
            merge_bunch_factor = self.initial_config["merge_bunching"]

        if "spacing" in self.initial_config:
            if self.initial_config["spacing"] == "gaussian":
                downscale = 5
                if "downscale" in self.initial_config:
                    downscale = self.initial_config["downscale"]
                startpositions, startlanes = self.gen_random_start_pos(downscale, bunch_factor, merge_bunch_factor, x0)
        else:
            startpositions, startlanes = self.gen_even_start_pos(bunch_factor, merge_bunch_factor, x0)

        return startpositions, startlanes

    def gen_even_start_pos(self, bunching, merge_bunching, x0=1):
        """
        Generate uniformly spaced start positions.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        startpositions = []
        startlanes = []

        # generate starting positions for non-merging vehicles
        # in order to avoid placing cars in the internal edges, their length is removed from the distribution length
        distribution_len = self.length - self.ring_0_n_len - self.ring_1_n_len
        increment = (distribution_len - bunching) * self.lanes_distribution / (self.num_vehicles - self.num_merge_vehicles)

        x = [x0] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_vehicles - self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            if ":ring_0" in pos[0]:
                x[lane_count] += self.ring_0_n_len
                pos = self.get_edge(x[lane_count])
            elif ":ring_1" in pos[0]:
                x[lane_count] += self.ring_1_n_len
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

        # generate starting positions for merging vehicles
        increment = (self.merge_in_len - merge_bunching) * self.lanes_distribution / self.num_merge_vehicles

        x = [self.get_x(edge="merge_in", position=0)] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] += increment

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes

    def gen_random_start_pos(self, downscale=5, bunching=0, merge_bunching=0, x0=1):
        """
        Generate random start positions via additive Gaussian.

        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.
        :return: list of start positions [(edge0, pos0), (edge1, pos1), ...]
                 list of start lanes
        """
        startpositions = []
        startlanes = []

        # generate starting positions for non-merging vehicles
        # in order to avoid placing cars in the internal edges, their length is removed from the distribution length
        distribution_len = self.length - self.ring_0_n_len - self.ring_1_n_len
        mean = (distribution_len - bunching) * self.lanes_distribution / (self.num_vehicles - self.num_merge_vehicles)

        x = [x0] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_vehicles - self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            if ":ring_0" in pos[0]:
                x[lane_count] += self.ring_0_n_len
                pos = self.get_edge(x[lane_count])
            elif ":ring_1" in pos[0]:
                x[lane_count] += self.ring_1_n_len
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

        # generate starting positions for merging vehicles
        mean = (self.merge_in_len - merge_bunching) * self.lanes_distribution / self.num_merge_vehicles

        x = [self.get_x(edge="merge_in", position=0)] * self.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] += np.random.normal(scale=mean / downscale, loc=mean)

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be distributed on in the network, reset
            if lane_count >= self.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes
