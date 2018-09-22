"""Contains the figure eight generator class."""

from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace
from lxml import etree

E = etree.Element


class Figure8Generator(Generator):
    """Generator for figure 8 lanes."""

    def __init__(self, net_params, base):
        """Instantiate the generator class."""
        super().__init__(net_params, base)

        r = net_params.additional_params["radius_ring"]
        lanes = net_params.additional_params["lanes"]
        ring_edgelen = r * pi / 2.
        intersection_edgelen = 2 * r
        self.name = "%s-%dm%dl" % (
            base, 2 * intersection_edgelen + 6 * ring_edgelen, lanes)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]

        nodes = [{
            "id": "center_intersection",
            "x": repr(0),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "top_upper_ring",
            "x": repr(r),
            "y": repr(2 * r),
            "type": "priority"
        }, {
            "id": "bottom_upper_ring_in",
            "x": repr(r),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "left_upper_ring",
            "x": repr(0),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "right_upper_ring",
            "x": repr(2 * r),
            "y": repr(r),
            "type": "priority"
        }, {
            "id": "top_lower_ring",
            "x": repr(-r),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "bottom_lower_ring",
            "x": repr(-r),
            "y": repr(-2 * r),
            "type": "priority"
        }, {
            "id": "left_lower_ring",
            "x": repr(-2 * r),
            "y": repr(-r),
            "type": "priority"
        }, {
            "id": "right_lower_ring_in",
            "x": repr(0),
            "y": repr(-r),
            "type": "priority"
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_edgelen = r * pi / 2.
        intersection_edgelen = 2 * r

        # intersection edges
        edges = [{
            "id": "right_lower_ring_in",
            "type": "edgeType",
            "priority": "78",
            "from": "right_lower_ring_in",
            "to": "center_intersection",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "right_lower_ring_out",
            "type": "edgeType",
            "priority": "78",
            "from": "center_intersection",
            "to": "left_upper_ring",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "bottom_upper_ring_in",
            "type": "edgeType",
            "priority": "46",
            "from": "bottom_upper_ring_in",
            "to": "center_intersection",
            "length": repr(intersection_edgelen / 2)
        }, {
            "id": "bottom_upper_ring_out",
            "type": "edgeType",
            "priority": "46",
            "from": "center_intersection",
            "to": "top_lower_ring",
            "length": repr(intersection_edgelen / 2)
        }]

        # ring edges
        edges += [{
            "id":
            "left_upper_ring",
            "type":
            "edgeType",
            "from":
            "left_upper_ring",
            "to":
            "top_upper_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 - cos(t)), r * (1 + sin(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top_upper_ring",
            "type":
            "edgeType",
            "from":
            "top_upper_ring",
            "to":
            "right_upper_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 + sin(t)), r * (1 + cos(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "right_upper_ring",
            "type":
            "edgeType",
            "from":
            "right_upper_ring",
            "to":
            "bottom_upper_ring_in",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * (1 + cos(t)), r * (1 - sin(t)))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top_lower_ring",
            "type":
            "edgeType",
            "from":
            "top_lower_ring",
            "to":
            "left_lower_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(pi / 2, pi, resolution)
            ])
        }, {
            "id":
            "left_lower_ring",
            "type":
            "edgeType",
            "from":
            "left_lower_ring",
            "to":
            "bottom_lower_ring",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(pi, 3 * pi / 2, resolution)
            ])
        }, {
            "id":
            "bottom_lower_ring",
            "type":
            "edgeType",
            "from":
            "bottom_lower_ring",
            "to":
            "right_lower_ring_in",
            "length":
            repr(ring_edgelen),
            "shape":
            " ".join([
                "%.2f,%.2f" % (-r + r * cos(t), -r + r * sin(t))
                for t in linspace(-pi / 2, 0, resolution)
            ])
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": repr(lanes),
            "speed": repr(speed_limit)
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "bottom_lower_ring": [
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring"
            ],
            "right_lower_ring_in": [
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring"
            ],
            "right_lower_ring_out": [
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in"
            ],
            "left_upper_ring": [
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out"
            ],
            "top_upper_ring": [
                "top_upper_ring", "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring"
            ],
            "right_upper_ring": [
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring"
            ],
            "bottom_upper_ring_in": [
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring"
            ],
            "bottom_upper_ring_out": [
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in"
            ],
            "top_lower_ring": [
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out"
            ],
            "left_lower_ring": [
                "left_lower_ring", "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring"
            ]
        }

        return rts

    def specify_connections(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        conn = []
        for i in range(lanes):
            conn += [{"from": "right_lower_ring_in",
                      "to": "right_lower_ring_out",
                      "fromLane": str(i),
                      "toLane": str(i)}]
            conn += [{"from": "bottom_upper_ring_in",
                      "to": "bottom_upper_ring_out",
                      "fromLane": str(i),
                      "toLane": str(i)}]
        return conn
