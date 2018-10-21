"""Contains the ring road generator class."""

from flow.core.generator import Generator

from numpy import pi, sin, cos, linspace, ceil, sqrt


class MultiCircleGenerator(Generator):
    """Generator for loop circle used in traffic simulation."""

    def __init__(self, net_params, base):
        """See parent class."""
        length = net_params.additional_params["length"]
        lanes = net_params.additional_params["lanes"]
        self.name = "%s-%dm%dl" % (base, length, lanes)

        super().__init__(net_params, base)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        ring_num = net_params.additional_params["num_rings"]

        r = length / (2 * pi)
        ring_spacing = 4*r
        num_rows = num_cols = int(ceil(sqrt(ring_num)))

        nodes = []
        i = 0
        for j in range(num_rows):
            for k in range(num_cols):
                nodes += [{
                    "id": "bottom_{}".format(i),
                    "x": repr(0 + j * ring_spacing),
                    "y": repr(-r + k * ring_spacing)
                }, {
                    "id": "right_{}".format(i),
                    "x": repr(r + j * ring_spacing),
                    "y": repr(0 + k * ring_spacing)
                }, {
                    "id": "top_{}".format(i),
                    "x": repr(0 + j * ring_spacing),
                    "y": repr(r + k * ring_spacing)
                }, {
                    "id": "left_{}".format(i),
                    "x": repr(-r + j * ring_spacing),
                    "y": repr(0 + k * ring_spacing)
                }]
                i += 1
                # FIXME this break if we don't have an exact square
                if i >= ring_num:
                    break
            if i >= ring_num:
                break

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        ring_num = net_params.additional_params["num_rings"]
        num_rows = num_cols = int(ceil(sqrt(ring_num)))
        r = length / (2 * pi)
        ring_spacing = 4 * r
        edgelen = length / 4.
        edges = []

        i = 0

        for j in range(num_rows):
            for k in range(num_cols):
                edges += [{
                    "id":
                    "bottom_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "bottom_{}".format(i),
                    "to":
                    "right_{}".format(i),
                    "length":
                    repr(edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * cos(t) + j * ring_spacing,
                                       r * sin(t) + k * ring_spacing)
                        for t in linspace(-pi / 2, 0, resolution)
                    ])
                }, {
                    "id":
                    "right_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "right_{}".format(i),
                    "to":
                    "top_{}".format(i),
                    "length":
                    repr(edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * cos(t) + j * ring_spacing,
                                       r * sin(t) + k * ring_spacing)
                        for t in linspace(0, pi / 2, resolution)
                    ])
                }, {
                    "id":
                    "top_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "top_{}".format(i),
                    "to":
                    "left_{}".format(i),
                    "length":
                    repr(edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * cos(t) + j * ring_spacing,
                                       r * sin(t) + k * ring_spacing)
                        for t in linspace(pi / 2, pi, resolution)
                    ])
                }, {
                    "id":
                    "left_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "left_{}".format(i),
                    "to":
                    "bottom_{}".format(i),
                    "length":
                    repr(edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * cos(t) + j * ring_spacing,
                                       r * sin(t) + k * ring_spacing)
                        for t in linspace(pi, 3 * pi / 2, resolution)
                    ])
                }]
                i += 1
                if i >= ring_num:
                    break
            if i >= ring_num:
                break

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
        ring_num = net_params.additional_params["num_rings"]
        rts = {}
        for i in range(ring_num):
            rts.update({
                "top_{}".format(i):
                    ["top_{}".format(i), "left_{}".format(i), "bottom_{}".format(i), "right_{}".format(i)],
                "left_{}".format(i): ["left_{}".format(i), "bottom_{}".format(i), "right_{}".format(i), "top_{}".format(i)],
                "bottom_{}".format(i): ["bottom_{}".format(i), "right_{}".format(i), "top_{}".format(i), "left_{}".format(i)],
                "right_{}".format(i): ["right_{}".format(i), "top_{}".format(i), "left_{}".format(i), "bottom_{}".format(i)]
            })

        return rts
