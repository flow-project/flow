import numpy as np

from cistar_dev.core.scenario import Scenario
from cistar_dev.scenarios.braess_paradox.gen import *


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

    def specify_edge_starts(self):
        """
        See parent class
        """
        edgestarts = \
            [("B",   0),
             ("BA1", self.curve_len),
             ("BA2", self.curve_len + self.horz_len),
             ("AC",  2 * self.curve_len + self.horz_len),
             ("CB",  2 * self.curve_len + self.horz_len + self.edge_len),
             ("AD",  2 * self.curve_len + self.horz_len + 2 * self.edge_len),
             ("DB",  2 * self.curve_len + self.horz_len + 3 * self.edge_len),
             ("CD",  2 * self.curve_len + self.horz_len + 4 * self.edge_len)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        intersection_edgestarts = \
            [(":A", 2 * self.curve_len + self.horz_len + 2 * self.edge_len + 0.1),
             (":B", 0 + 0.1),
             (":C", 2 * self.curve_len + self.horz_len + 0.1),
             (":D", 2 * self.curve_len + self.horz_len + 3 * self.edge_len + 0.1)]

        return intersection_edgestarts
