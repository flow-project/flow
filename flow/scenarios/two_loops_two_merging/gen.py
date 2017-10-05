from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace


class TwoLoopTwoMergingGenerator(Generator):
    """
    Generator for a two-loop network in which both loops merge into a common lane. Requires from net_params:
     - ring_radius: radius of the ring roads
     - lanes: number of lanes in the network
     - speed_limit: max speed limit in the network
     - resolution: number of nodes resolution
    """

    def __init__(self, net_params, net_path, cfg_path, base):
        """
        See parent class
        """
        radius = net_params["ring_radius"]
        lanes = net_params["lanes"]

        super().__init__(net_params, net_path, cfg_path, base)

        self.name = "%s-%dr%dl" % (base, radius, lanes)

    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params["ring_radius"]
        angle = pi/3

        nodes = [{"id": "bottom", "x": repr(0),                     "y": repr(-r * sin(angle))},
                 {"id": "right",  "x": repr(r * (1 + cos(angle))),  "y": repr(0)},
                 {"id": "top",    "x": repr(0),                     "y": repr(r * sin(angle))},
                 {"id": "left",   "x": repr(-r * (1 + cos(angle))), "y": repr(0)}]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params["ring_radius"]
        resolution = net_params["resolution"]

        angle = pi/3
        merge_len = 2 * r * sin(angle)
        ring_edgelen = 2 / 3 * pi * r

        edges = [{"id": "merge", "type": "edgeType",
                  "from": "bottom", "to": "top", "length": repr(merge_len)},

                 # TODO: what's wrong with the look? Is it affecting anything?
                 {"id": "right_top", "type": "edgeType",
                  "from": "top", "to": "right", "length": repr(ring_edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (1.08*r * (cos(t) + cos(angle)), 1.08*r * sin(t))
                                     for t in linspace(2 / 3 * pi, 0, resolution)])},

                 {"id": "right_bottom", "type": "edgeType",
                  "from": "right", "to": "bottom", "length": repr(ring_edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (1.08*r * (cos(t) + cos(angle)), 1.08*r * sin(t))
                                     for t in linspace(0, - 2/3 * pi, resolution)])},

                 {"id": "left_top", "type": "edgeType",
                  "from": "top", "to": "left", "length": repr(ring_edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * (cos(t) - cos(angle)), r * sin(t))
                                     for t in linspace(pi / 3, pi, resolution)])},

                 {"id": "left_bottom", "type": "edgeType",
                  "from": "left", "to": "bottom", "length": repr(ring_edgelen),
                  "shape": " ".join(["%.2f,%.2f" % (r * (cos(t) - cos(angle)), r * sin(t))
                                     for t in linspace(pi, 5 / 3 * pi, resolution)])}]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        lanes = net_params["lanes"]
        speed_limit = net_params["speed_limit"]
        types = [{"id": "edgeType", "numLanes": repr(lanes), "speed": repr(speed_limit)}]

        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"right_top": ["right_top", "right_bottom", "merge", "right_top"],
               "right_bottom": ["right_bottom", "merge", "right_top", "right_bottom"],
               "left_top": ["left_top", "left_bottom", "merge", "left_top"],
               "left_bottom": ["left_bottom", "merge", "left_top", "left_bottom"]}

        return rts
