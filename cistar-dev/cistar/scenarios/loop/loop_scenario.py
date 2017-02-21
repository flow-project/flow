from cistar.core.scenario import Scenario
from cistar.core.generator import Generator

import logging
import numpy as np

class LoopScenario(Scenario):

    def __init__(self, name, num_vehicles, type_params, cfg_params, net_params, initial_config=None, cfg=None,
                 generator_class=None):
        super().__init__(name, num_vehicles, type_params, cfg_params, net_params, initial_config, cfg, generator_class)

        if "length" not in net_params:
            raise ValueError("length of circle not supplied")
        else:
            self.length = net_params["length"]

        if "lanes" not in net_params:
            raise ValueError("lanes of circle not supplied")
        else:
            self.lanes = net_params["lanes"]

        if "speed_limit" not in net_params:
            raise ValueError("speed limit of circle not supplied")
        else:
            self.speed_limit = net_params["speed_limit"]

        if "resolution" not in net_params:
            raise ValueError("resolution of circle not supplied")
        else:
            self.resolution = net_params["resolution"]

        edgelen = self.length/4
        self.edgestarts = [("bottom", 0),("right", edgelen), ("top", 2*edgelen),( "left", 3*edgelen)]

        if "positions" not in self.initial_config:
            self.initial_config["positions"] = self.gen_random_start_pos()
        # self.initial_config["shuffle"] = True
        if not cfg:
            self.cfg = self.generate()

    def get_edge(self, x):
        for (e, s) in self.edgestarts:
            if x >= s:
                starte = e
                startx = x - s
        return starte, startx

    def get_x(self, edge, position):
        for edge_tuple in self.edgestarts:
            if edge_tuple[0] == edge:
                edge_start = edge_tuple[1]
                break
        return position + edge_start

    def gen_even_start_positions(self):
        startpositions = []
        increment = self.length/self.num_vehicles

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