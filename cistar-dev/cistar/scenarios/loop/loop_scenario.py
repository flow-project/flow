from cistar.core.scenario import Scenario
from cistar.scenarios.loop.gen import CircleGenerator

import numpy as np

class LoopScenario(Scenario):

    def __init__(self, name, type_params, net_params, cfg_params, initial_config=None, cfg=None):
        """
        Initialize a loop scenario.
        :param net_params:
            Must include: length, lanes, speed_limit_resolution
        :param initial_config:
            Can include: positions, shuffle
        :param generator_class:
        """
        super().__init__(name, type_params, net_params, cfg_params, initial_config, cfg, CircleGenerator)

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

        edgelen = self.length/4
        self.edgestarts = [("bottom", 0),("right", edgelen), ("top", 2*edgelen),( "left", 3*edgelen)]

        if "positions" not in self.initial_config:
            self.initial_config["positions"] = self.gen_even_start_positions()
        self.initial_config["shuffle"] = True
        if not cfg:
            self.cfg, self.outs = self.generate()

    def get_edge(self, x):
        starte = ""
        startx = 0
        for (e, s) in self.edgestarts:
            if x >= s:
                starte = e
                startx = x - s
        return starte, startx

    def get_x(self, edge, position):
        edge_start = 0
        for edge_tuple in self.edgestarts:
            if edge_tuple[0] == edge:
                edge_start = edge_tuple[1]
                break
        return position + edge_start

    def gen_even_start_positions(self):
        startpositions = []
        increment = (self.length-10)/self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            pos = self.get_edge(x)
            startpositions.append(pos)
            x += increment

        return startpositions

    def gen_random_start_pos(self):
        startpositions = []
        mean = self.length/self.num_vehicles

        x = 1
        for i in range(self.num_vehicles):
            pos = self.get_edge(x)
            startpositions.append(pos)
            x += np.random.normal(scale=mean/5, loc=mean)

        return startpositions
