"""Contains the loop merge generator class."""

from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace


class TwoLoopOneMergingGenerator(Generator):
    """Generator for a two loop merge network.

    This network consists of two loops that both merge into a common lane.
    """

    def __init__(self, net_params, base):
        """See parent class."""
        radius = net_params.additional_params["ring_radius"]
        self.inner_lanes = net_params.additional_params["inner_lanes"]
        self.outer_lanes = net_params.additional_params["outer_lanes"]

        super().__init__(net_params, base)

        self.name = "%s-%dr%dl" % (base, radius,
                                   self.inner_lanes + self.outer_lanes)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        nodes = [{
            "id": "top_left",
            "x": repr(0),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "bottom_left",
            "x": repr(0),
            "y": repr(-r),
            "type": "priority"
        }, {
            "id": "top_right",
            "x": repr(x),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "bottom_right",
            "x": repr(x),
            "y": repr(-r),
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        ring_edgelen = pi * r
        resolution = 40

        edges = [{
            "id":
            "center",
            "from":
            "bottom_left",
            "to":
            "top_left",
            "type":
            "edgeType",
            "length":
            repr(ring_edgelen),
            "priority":
            "46",
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, pi / 2, resolution)
            ]),
            "numLanes":
            str(self.inner_lanes)
        }, {
            "id": "top",
            "from": "top_right",
            "to": "top_left",
            "type": "edgeType",
            "length": repr(x),
            "priority": "46",
            "numLanes": str(self.outer_lanes)
        }, {
            "id": "bottom",
            "from": "bottom_left",
            "to": "bottom_right",
            "type": "edgeType",
            "length": repr(x),
            "numLanes": str(self.outer_lanes)
        }, {
            "id":
            "left",
            "from":
            "top_left",
            "to":
            "bottom_left",
            "type":
            "edgeType",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(pi / 2, 3 * pi / 2, resolution)
            ]),
            "numLanes":
            str(self.inner_lanes)
        }, {
            "id":
            "right",
            "from":
            "bottom_right",
            "to":
            "top_right",
            "type":
            "edgeType",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (x + r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, pi / 2, resolution)
            ]),
            "numLanes":
            str(self.outer_lanes)
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{"id": "edgeType", "speed": repr(speed_limit)}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "top": ["top", "left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left", "bottom"],
            "right": ["right", "top", "left", "bottom"],
            "left": ["left", "center", "left"],
            "center": ["center", "left", "center"]
        }

        return rts
