from flow.scenarios.base_scenario import Scenario

from numpy import pi, sin, cos


class TwoLoopsTwoMergingScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=None):
        """
        Initializes a loop scenario. Required net_params: length, lanes,
        speed_limit, resolution. Required initial_config: positions.

        See Scenario.py for description of params.
        """
        radius = net_params.additional_params["ring_radius"]
        net_params.additional_params["length"] = 8 / 3 * pi * radius + 2 * radius * sin(pi / 3)

        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config=initial_config)

    def specify_edge_starts(self):
        """
        See parent class
        """
        r = self.net_params.additional_params["ring_radius"]
        ring_edgelen = 2 / 3 * pi * r

        edgestarts = [("right_top", 0),
                      ("right_bottom", ring_edgelen + 0.3),
                      ("left_top", 2 * ring_edgelen + 7.3),
                      ("left_bottom", 3 * ring_edgelen + 7.6),
                      ("merge", 4 * ring_edgelen + 14.6)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See base class
        """
        r = self.net_params.additional_params["ring_radius"]
        ring_edgelen = 2 / 3 * pi * r

        internal_edgestarts = \
            [(":right", ring_edgelen),
             (":bottom", 2 * ring_edgelen + 0.3),
             (":left", 3 * ring_edgelen + 7.3),
             (":top", 4 * ring_edgelen + 7.6)]

        return internal_edgestarts
